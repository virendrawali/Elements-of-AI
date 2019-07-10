from dataset import *
from lib import pca
import numpy as np

train = read_data("train-data.txt")
X = train.pixel_values

def compute_reconstruction_error(X, n_components=10):
    X_mean, V = pca(X, n_components)

    x = np.dot(X-X_mean, V)
    
    x = np.dot(x, V.T) + X_mean
    

    return (np.sqrt(((x-X)**2).sum()))


n_components = list(range(10, 192, 20))
errors = [compute_reconstruction_error(X, n) for n in n_components]

import matplotlib.pyplot as plt 
plt.plot(n_components, errors, "-x")
plt.xlabel("Number of PCA components")
plt.ylabel("Reconstruction error")
plt.show()