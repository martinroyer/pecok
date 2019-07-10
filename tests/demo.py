import numpy as np

np.random.seed(232123123)
n = 10
p = 100
X = np.zeros((n, p))
snr = 0.3
X[n//2:, :] = -np.ones(p) * snr + np.random.normal(scale=.1, size=(n//2,p))
X[:n//2, :] = np.ones(p) * snr + np.random.normal(scale=1, size=(n//2,p))

from sklearn.cluster import KMeans
print(KMeans(n_clusters=2, init='k-means++', n_init=100).fit(X).labels_)
from pecok import Pecok
print(Pecok(n_clusters=2, corr=0).fit(X).labels_)
print(Pecok(n_clusters=2, corr=4).fit(X).labels_)
