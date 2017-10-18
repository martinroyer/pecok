import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn import cluster
from pecok import pecok_clustering

seed = 432
np.random.seed(seed)
print("seed is %i" % seed)

n_samples = 10
n_features = 100
truth = np.concatenate((np.repeat(0, n_samples//2), np.repeat(1, n_samples//2)))
X = np.zeros((n_samples, n_features))
X[:n_samples//2, :] = np.ones(n_features)*0.1 + np.random.normal(scale=1, size=(n_samples//2, n_features))
X[n_samples//2:, :] = -np.ones(n_features)*0.1 + np.random.normal(scale=0.1, size=(n_samples//2, n_features))

Bhat = pecok_clustering.cluster(X, 2)
kMeans = cluster.KMeans(n_clusters=2, init='k-means++', n_init=100, copy_x=True)
print("truth:".ljust(10), truth)
print("pecok:".ljust(10), kMeans.fit(Bhat).labels_)
print("kmeans:".ljust(10), kMeans.fit(X).labels_)
