import math
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras import layers
from scipy.signal.windows import tukey
from tensorflow.keras import mixed_precision

import numpy as np

embedding_dims = 32
embedding_max_frequency = 1000.0

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        ))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=2)
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[2]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv1D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv1D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv1D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling1D(size=2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def ContextAttention(num_heads=9, dropout=0.0):
    def apply(x):
        x, c = x
        conv = layers.Conv1D(x.shape[2], kernel_size=1, padding='same')
        return layers.MultiHeadAttention(num_heads=num_heads, key_dim=2, dropout=dropout)(x,
                                                                                          conv(c),
                                                                                          return_attention_scores=False)
    return apply

def get_network(image_size, context_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, 3))
    noise_variances = keras.Input(shape=(1, 1))
    context = keras.Input((1, context_size))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling1D(size=image_size)(e)

    x = layers.Conv1D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    #Injecting context
    x = layers.Concatenate()([x, layers.UpSampling1D(x.shape[1])(context)])

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv1D(3, kernel_size=1, kernel_initializer='zeros')(x)

    return keras.Model([noisy_images, noise_variances, context], x, name="residual_unet")

class DiffusionModel(keras.Model):
    def __init__(self,
                 batch_size,
                 image_size,
                 context_size,
                 widths,
                 block_depth,
                 min_signal_rate=0.01,
                 max_signal_rate=0.95,
                 ema=0.999):
        super().__init__()

        # sampling
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.ema = ema
        self.batch_size = batch_size
        self.image_size = image_size
        self.network = get_network(image_size, context_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.total_loss_tracker]

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, context, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2, context], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, context):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, context, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps, context):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, self.image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, context)
        return generated_images

    def train_step(self, images):
        if isinstance(images, tuple):
            images, metadata = images

        noises = tf.random.normal(shape=(self.batch_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, metadata, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.total_loss_tracker.update_state(noise_loss + image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        if isinstance(images, tuple):
            images, metadata = images

        noises = tf.random.normal(shape=(self.batch_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, metadata, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        self.total_loss_tracker.update_state(noise_loss + image_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        #images = self.denormalize(images)
        #generated_images = self.generate(
        #    num_images=batch_size, diffusion_steps=kid_diffusion_steps
        #)
        #self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=5, context=None, dataset='instance'):
        # plot random generated images for visual evaluation of generation quality

        generated_images = self.generate(
            num_images=num_rows,
            diffusion_steps=20,
            context=context,
        )

        generated_images *= tukey(generated_images.shape[1], alpha=0.05)[np.newaxis,:,np.newaxis]

        fig, axs = plt.subplots(nrows=num_rows, ncols=1)
        for row in range(num_rows):
            axs[row].plot(generated_images[row,:,0])
        plt.tight_layout()
        plt.savefig(f'tf/outputs/{dataset}_epoch={epoch}.png')
        plt.close()
