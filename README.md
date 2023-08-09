# Stable Diffusion for Generation of Earthquakes
Project for UKEN 2022. 
Using stable diffusion to generate events from metadata. Can also be used in reverse to denoise waveforms (not done yet, just an idea). 

# Files 
## model.py
Defining the stable diffusion model. Adapted from somewhere...

## train.py
Training of the stabel diffusion model. 
Summary:
- Loading the STEAD dataset.
- Defining which metadata columns to use for generation.
- Define hyperparameters for models.
- Create and compile model.
- Train model and plot examples along the way. 
