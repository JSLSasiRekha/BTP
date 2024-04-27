import scipy.io
import numpy as np

# Load the .mat file
data = scipy.io.loadmat('pure3D_labels.mat')  # Replace 'your_dataset.mat' with your file path

# Extract the hyperspectral cube from the loaded data
cube = data['labels']  # Adjust the key depending on how the data is structured in your .mat file

# Get the dimensions of the dataset
rows, cols, bands = cube.shape

# Reshape the 3D dataset into 2D (flattening each pixel's spectral information)
cube_2d = np.reshape(cube, (rows * cols, bands))

# Save the 2D dataset into a new .mat file
scipy.io.savemat('pure_labels.mat', {'cube_2d': cube_2d})
