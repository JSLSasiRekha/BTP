import scipy.io

# Load MATLAB file
mat = scipy.io.loadmat('mixed_labels.mat')

# List all variables in the MATLAB file
print("Variables in the MATLAB file:")
print(mat.keys())

# Access specific variable
# For example, if the variable is named 'data':
data = mat['cube_2d']

# Display data
print("Shape of 'data' variable:", data.shape)
print("Data:")
print(data)
