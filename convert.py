from scipy.io import loadmat

# Load the predicted pure spectral bands from the .mat file
predicted_pure_data = loadmat('predicted_pure_spectral_bands.mat')
predicted_pure_spectral_bands = predicted_pure_data['predicted_pure_spectral_bands']

# Reshape the predicted pure spectral bands
reshaped_predicted_pure_spectral_bands = predicted_pure_spectral_bands.reshape(predicted_pure_spectral_bands.shape[0], -1)

# Print the new shape
print("New shape of predicted pure spectral bands:", reshaped_predicted_pure_spectral_bands.shape)
