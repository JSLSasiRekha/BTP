import numpy as np
import scipy.io as sio
from spectral import *

# Load Cuprite dataset
cuprite_data = sio.loadmat('pureTrain.mat')['pure']

# Load reference spectrum
reference_spectrum = sio.loadmat('groundtruth.mat')['M']

print("Shape of cuprite_data:", cuprite_data.shape)
print("Shape of reference_spectrum:", reference_spectrum.shape)

# Transpose reference_spectrum to match expected shape
reference_spectrum = reference_spectrum.T

print("Shape of transposed reference_spectrum:", reference_spectrum.shape)

# Reshape data for SAM calculation
n_rows, n_cols, n_bands = cuprite_data.shape
data_reshaped = cuprite_data.reshape((n_rows * n_cols, n_bands))

print("Shape of data_reshaped before transpose:", data_reshaped.shape)

# Transpose data_reshaped to match expected shape
data_reshaped = data_reshaped.T

print("Shape of data_reshaped after transpose:", data_reshaped.shape)

# Compute Spectral Angle Mapper (SAM)
sam_map = spectral_angles(cuprite_data, reference_spectrum)

# Reshape SAM map to original image dimensions
# sam_map_reshaped = sam_map.reshape((n_rows, n_cols))

# Threshold SAM map to generate labels
threshold = 0.8  # Adjust as needed
labels = sam_map <= threshold

# Save labels to a .mat file
sio.savemat('pure_labels.mat', {'labels': labels})

print('Labels saved to cuprite_labels.mat')
