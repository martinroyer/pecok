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
p,q = 0.7, 0.3
truth = np.concatenate((np.repeat(0, n_samples//2), np.repeat(1, n_samples//2)))
expA = q*np.ones((n_samples, n_samples))
expA[:n_samples//2,:n_samples//2] += p-q
expA[n_samples//2:,n_samples//2:] += p-q
A = np.vectorize(lambda p : np.random.binomial(1, p))(expA)
A.flat[::n_samples+1] = 0
i_lower = np.tril_indices(n_samples, -1)
A[i_lower] = A.T[i_lower]
print(A)

Bhat = pecok_clustering.cluster_sbm(A, 2)
kMeans = cluster.KMeans(n_clusters=2, init='k-means++', n_init=100, copy_x=True)
print("truth:".ljust(10), truth)
print("pecok:".ljust(10), kMeans.fit(Bhat).labels_)
print("kmeans:".ljust(10), kMeans.fit(A).labels_)
