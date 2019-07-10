from math import sqrt, log
from collections import namedtuple, Counter
import numpy.linalg as linalg
import numpy as np

def euclidean_distance(x, X):
    return np.sum(X**2, axis=1) + np.sum(x**2, axis=1)[:, np.newaxis] - 2 * np.dot(x, X.T) 

def cosine_distance(x, X):
    numerator = np.dot(x, X.T)
    denominator = linalg.norm(x, axis=1) * linalg.norm(X, axis=1)
    return 1 - numerator / denominator


def entropy(ls):
    counts = Counter(ls)
    total = len(ls)
    ps = [count/total for key, count in counts.items()]
    return sum(-p*log(p) for p in ps)


Tree = namedtuple("Tree", ["key", "value", "left", "right"])
def build_tree(X, y, max_depth):
    pass

def pca(X, n_components=10):
    X = np.array(X)
    X_mean = X.mean(axis=0) # mean over each column
    X = X-X_mean # center each column
    U, S, VT = linalg.svd(X, full_matrices=False)
    V = VT.T 
    return X_mean, V[:, :n_components]

