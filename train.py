

import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesResampler
from scipy.signal.windows import tukey
from tqdm import tqdm
import sys
sys.path.append('/tf')

from model import DiffusionModel
import tensorflow as tf
import tensorflow_addons as tfa

sample_rate = 100
nyq = 0.5 * sample_rate

TIMESERIES_SIZE = 6000

from scipy.signal import butter, sosfiltfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    lowcut = lowcut / nyq
    highcut = highcut / nyq
    return butter(order, [lowcut, highcut], analog=False, btype='band', output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfiltfilt(sos, data, axis=1)

def load_data(filename, metadata):
    X = []
    with h5py.File(filename, 'r') as f:
        for trace_name in tqdm(metadata['trace_name'].values):
            a = np.array(f['data'][trace_name], dtype='float32')
            X.append(a)
    return np.swapaxes(np.stack(X, axis=0), 1, 2)

from obspy.geodetics.base import gps2dist_azimuth


import numpy.ma as ma

def normalize(X, mode='max', channel_mode='local'):
    X -= np.mean(X, axis=0, keepdims=True)

    if mode == 'max':
        if channel_mode == 'local':
            m = np.max(X, axis=0, keepdims=True)
        else:
            m = np.max(X, keepdims=True)
    elif mode == 'std':
        if channel_mode == 'local':
            m = np.std(X, axis=0, keepdims=True)
        else:
            m = np.std(X, keepdims=True)
    else:
        raise NotImplementedError(
            f'Not supported normalization mode: {mode}')

    m[m == 0] = 1
    return X / m

class STEADGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 filename,
                 metadata_filename,
                 batch_size=32,
                 shuffle=True):
        self.data = h5py.File(filename, 'r')
        self.metadata = pd.read_csv(metadata_filename, low_memory=False, na_values=['nan', 'None'])
        self.metadata = self.metadata[self.metadata.trace_category == 'earthquake_local']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.events = self.metadata['trace_name'].to_list()
        metadata_columns = ['source_depth_km',
                            'source_magnitude',
                            'receiver_elevation_m',
                            'dist',
                            'azi_station_source_a',
                            'azi_station_source_b',
                            'azi_source_station_a',
                            'azi_source_station_b']
        self.metadata = self.encode_metadata(self.metadata)
        self.y = self.metadata[metadata_columns].values
        self.scaler = StandardScaler()
        self.y = self.scaler.fit_transform(self.y)
        self.y = np.where(np.isnan(self.y), ma.array(self.y, mask=np.isnan(self.y)) * self.scaler.mean_, self.y)
        self.on_epoch_end()

    def encode_metadata(self, metadata):
        source_lat = metadata['source_latitude'].values
        source_long = metadata['source_longitude'].values
        station_lat = metadata['receiver_latitude'].values
        station_long = metadata['receiver_longitude'].values
        dist, azi_source_station, azi_station_source = zip(
            *list(map(lambda x: gps2dist_azimuth(*x), zip(source_lat, source_long, station_lat, station_long))))

        azi_source_station = list(map(lambda x: x * (np.pi / 180.0), azi_source_station))
        azi_station_source = list(map(lambda x: x * (np.pi / 180.0), azi_station_source))

        metadata['dist'] = np.asarray(dist) / 1000.0
        metadata['azi_source_station_a'] = np.cos(azi_source_station)
        metadata['azi_source_station_b'] = np.sin(azi_source_station)

        metadata['azi_station_source_a'] = np.cos(azi_station_source)
        metadata['azi_station_source_b'] = np.sin(azi_station_source)

        return metadata

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def getsingle(self, trace_name):
        return np.array(self.data.get('data/'+str(trace_name)))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        events = [self.events[i] for i in indexes]
        x, y = np.stack([self.getsingle(event) for event in events], axis=0), np.expand_dims(self.y[indexes], 1)
        return x.astype('float32'), y.astype('float32')

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.events))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def encode_metadata(metadata):
    source_lat = metadata['source_latitude_deg'].values
    source_long = metadata['source_longitude_deg'].values
    station_lat = metadata['station_latitude_deg'].values
    station_long = metadata['station_longitude_deg'].values
    dist, azi_source_station, azi_station_source = zip(*list(map(lambda x: gps2dist_azimuth(*x), zip(source_lat, source_long, station_lat, station_long))))

    azi_source_station = list(map(lambda x: x * (np.pi / 180.0), azi_source_station))
    azi_station_source = list(map(lambda x: x * (np.pi / 180.0), azi_station_source))

    metadata['dist'] = dist / 1000.0
    metadata['azi_source_station_a'] = np.cos(azi_source_station)
    metadata['azi_source_station_b'] = np.sin(azi_source_station)

    metadata['azi_station_source_a'] = np.cos(azi_station_source)
    metadata['azi_station_source_b'] = np.sin(azi_station_source)

    return metadata

class Generator(tf.keras.utils.Sequence):
    def __init__(self,
                 files,
                 batch_size=32,
                 bandpass=(2.0, 8.0),
                 num_files_per_epoch=1,
                 resampler=None,
                 shuffle=True):

        self.files = files
        self.resampler = resampler
        self.num_files_per_epoch  = num_files_per_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bandpass = bandpass
        self.on_epoch_end()

    def load_file(self, filename):
        with np.load(filename) as a:
            x = a['x']
            y = a['y']

        return x, y

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item*self.batch_size:(item+1)*self.batch_size]
        x, y = self.X[indexes], self.y[indexes]
        x *= tukey(x.shape[1], alpha=0.05)[np.newaxis,:,np.newaxis]
        if not self.bandpass is None:
            x = butter_bandpass_filter(x, self.bandpass[0], self.bandpass[1], fs=sample_rate)
        x = np.asarray(list(map(lambda a: normalize(a, mode='max', channel_mode='global'), x)))
        if not self.resampler is None:
            x = self.resampler.fit_transform(x)

        y[np.isnan(y)] = 0

        return x, y

    def on_epoch_end(self):
        self.X, self.y = [], []
        for _ in range(self.num_files_per_epoch):
            X, y = self.load_file(np.random.choice(self.files))
            self.X.append(X)
            self.y.append(y)
        if self.num_files_per_epoch > 1:
            self.X = np.concatenate(self.X, axis=0)
            self.y = np.concatenate(self.y, axis=0)
        else:
            self.X = self.X[0]
            self.y = self.y[0]

        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)

preprocess = False

path = '/data'
n_splits = 100

if preprocess:
    metadata_columns = ['source_depth_km',
                        'source_magnitude',
                        'station_elevation_m',
                        'dist',
                        'azi_station_source_a',
                        'azi_station_source_b',
                        'azi_source_station_a',
                        'azi_source_station_b']

    scaler = StandardScaler()
    metadata = pd.read_csv(f'{path}/metadata/metadata_Instance_events.csv', low_memory=False).sample(frac=1.0)
    metadata = encode_metadata(metadata)
    scaler.fit(metadata[metadata_columns].values)

    metadatas = np.array_split(metadata, n_splits)
    for i, df in enumerate(tqdm(metadatas)):
        X = load_data(f'{path}/data/Instance_events_counts.hdf5', df)
        y = scaler.transform(df[metadata_columns].values)
        y = np.where(np.isnan(y), ma.array(y, mask=np.isnan(y)) * scaler.mean_, y)
        np.savez(f'{path}/{i}.npz', x=X, y=y)

batch_size = 128
#files = [f'{path}/{i}.npz' for i in range(n_splits)]

import glob
files = sorted(list(glob.glob(f'{path}/stead_*.npz')))
num_files = len(files)

data = Generator(files[:-1],
                 batch_size=batch_size,
                 bandpass=None,
                 num_files_per_epoch=20)
val_data = Generator(files[-1:],
                     batch_size=batch_size,
                     bandpass=None,
                     num_files_per_epoch=1)

#data = STEADGenerator(filename='/data/merge.hdf5',
#                      metadata_filename='/data/merge.csv',
#                      batch_size=4096,
#                      shuffle=False)

#for i, (x, y) in tqdm(enumerate(data), total=len(data)):
#    np.savez(f'/data/stead_{i}.npz',
#         x=np.asarray(x, dtype='float32'),
#         y=np.asarray(y, dtype='float32'))

# data
num_epochs = 50 * (num_files // data.num_files_per_epoch) # train for at least 50 epochs for good results

# architecture
widths = [32, 64, 96, 128]
context_size = data.y.shape[-1]
data_size = data.X.shape[1]
block_depth = 2

# optimization
learning_rate = 1e-3
weight_decay = 1e-4

model = DiffusionModel(batch_size,
                       data_size,
                       context_size,
                       widths,
                       block_depth)
# below tensorflow 2.9:
# pip install tensorflow_addons
# import tensorflow_addons as tfa
# optimizer=tfa.optimizers.AdamW
model.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=1
    ),
    loss=tf.keras.losses.mean_absolute_error,
)

model.network.summary()
checkpoint_path = "checkpoints/diffusion_model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="i_loss",
    mode="min",
    save_best_only=True,
)

num_rows = 5
# run training and plot generated images periodically
model.fit(
    data,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=val_data,
    validation_freq=10,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: model.plot_images(epoch, logs,
                                                                                   num_rows=num_rows,
                                                                                   dataset='stead',
                                                                                   context=val_data.y[:num_rows])),
        checkpoint_callback,
    ],
)
