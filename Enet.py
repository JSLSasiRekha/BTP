import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import DictionaryLearning
from scipy.io import loadmat
from scipy.io import savemat

# Load the hyperspectral image dataset
hsi_data = loadmat('./TNNLS_Data/cuprite/cuprite.mat')
hsi_cube = hsi_data['X']  # Assuming the dataset is named 'cube_3d'

# Calculate K based on 20% of the hyperspectral pixels
K_percentage = 2 / 100
total_pixels = hsi_cube.shape[0] * hsi_cube.shape[1]
K = 224
print("K:", K)

# Step 1: Divide the HSI into partially overlapped blocks
def divide_into_blocks(hsi, block_size, overlap_ratio):
    blocks = []
    for i in range(0, hsi.shape[0] - block_size + 1, block_size // 2):
        for j in range(0, hsi.shape[1] - block_size + 1, block_size // 2):
            block = hsi[i:i+block_size, j:j+block_size, :]
            blocks.append(block)
    return np.array(blocks)

# Step 2: Apply VCA to extract endmembers from each block
def vca_extraction(block):
    num_endmembers = 12  # Automatically estimated by HySime or similar method
    pca = PCA(n_components=num_endmembers, whiten=True)
    pca.fit(block.reshape(-1, block.shape[2]))
    endmembers = pca.components_.T
    return endmembers

# Step 3: Apply K-Means clustering to remove repeated endmembers and aggregate them into K clusters
def clustering(endmembers):
    kmeans = KMeans(n_clusters=K, n_init=10)  # Set the value of n_init explicitly
    labels = kmeans.fit_predict(endmembers)
    unique_labels = np.unique(labels)
    aggregated_endmembers = np.zeros((len(unique_labels), endmembers.shape[1]))
    for i, label in enumerate(unique_labels):
        aggregated_endmembers[i] = np.mean(endmembers[labels == label], axis=0)
    return aggregated_endmembers

# Extract spectral bundles using the above functions
def extract_spectral_bundles(hsi, block_size):
    blocks = divide_into_blocks(hsi, block_size, overlap_ratio=0.5)
    print("Number of blocks:", blocks.shape)
    spectral_bundles = []
    i=0
    for block in blocks:
        if(i%100==0):
            print(i)    
        endmembers = vca_extraction(block)
        # print("Shape of endmembers:", endmembers.shape)
        aggregated_endmembers = clustering(endmembers)
        # print("Shape of aggregated endmembers:", aggregated_endmembers.shape)
        spectral_bundles.append(aggregated_endmembers)
        i=i+1
    return np.array(spectral_bundles)

# Predict pure spectral bands using dictionary learning
def predict_pure_spectral_bands(spectral_bundles):
    print("Shape of spectral bundles:", spectral_bundles.shape)
    pure_spectral_bands = []
    for spectral_bundle in spectral_bundles:
        dl = DictionaryLearning(n_components=spectral_bundle.shape[0], alpha=1, max_iter=1000)
        dl.fit(spectral_bundle.T)
        pure_spectral_band = dl.components_
        pure_spectral_bands.append(pure_spectral_band)
    return np.array(pure_spectral_bands)

# Parameters
block_size = 50  # Size of each block

# Perform the steps to extract spectral bundles and predict pure spectral bands
spectral_bundles = extract_spectral_bundles(hsi_cube, block_size)
print("Shape of spectral bundles:", spectral_bundles.shape)
predicted_pure_spectral_bands = predict_pure_spectral_bands(spectral_bundles)

# Reshape the predicted pure spectral bands
num_bands = predicted_pure_spectral_bands.shape[0]
reshaped_predicted_pure_spectral_bands = predicted_pure_spectral_bands.reshape(num_bands, -1)

# Print the shape of the reshaped predicted pure spectral bands
print("New shape of reshaped predicted pure spectral bands:", reshaped_predicted_pure_spectral_bands.shape)

# Save the reshaped predicted pure spectral bands
savemat('predicted_pure_spectral_bands.mat', {'predicted_pure_spectral_bands': reshaped_predicted_pure_spectral_bands})
