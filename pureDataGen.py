import numpy as np
import scipy.io

# Load Cuprite dataset and endmember ground truth from .mat files
cuprite_data = scipy.io.loadmat('cuprite.mat')['X']
endmembers = scipy.io.loadmat('groundtruth.mat')['M']

# Define threshold for purity classification
threshold = 0.7  # Adjust as needed

# Reshape endmembers array to match cuprite_data shape
endmembers = endmembers.T  # Transpose to match cuprite_data shape

# Function to classify pixels to the closest mineral and return its spectrum
def classify_mineral(pixel_spectrum):
    distances = np.linalg.norm(pixel_spectrum - endmembers, axis=1)
    closest_mineral_index = np.argmin(distances)
    return endmembers[closest_mineral_index]

# Classify pixels in Cuprite dataset
classified_pixels = np.zeros((cuprite_data.shape[0], cuprite_data.shape[1], cuprite_data.shape[2]))

for i in range(cuprite_data.shape[0]):
    for j in range(cuprite_data.shape[1]):
        pixel_spectrum = cuprite_data[i, j, :]
        closest_mineral_spectrum = classify_mineral(pixel_spectrum)
        classified_pixels[i, j, :] = closest_mineral_spectrum

# Save classified pixels to .mat file
scipy.io.savemat('pureTrain.mat', {'pure': classified_pixels})
