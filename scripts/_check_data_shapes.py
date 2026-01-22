import numpy as np

X = np.load('data/processed/node_feature_tensor.npy')
A = np.load('data/processed/adjacency_matrix.npy')

print('X.shape =', X.shape)
print('A.shape =', A.shape)
