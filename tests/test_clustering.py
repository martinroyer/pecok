"""Test for variable and point clustering"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

import numpy as np
from sklearn import cluster
import scipy.sparse.linalg as ssl

from pecok import gamma, admm


def hierarchial_clustering(obs, n_struct):
    hclustering = cluster.AgglomerativeClustering(linkage='ward', n_clusters=n_struct)
    return hclustering.fit(obs, n_struct)


def kmeans_clustering(obs, n_struct):
    k_means = cluster.KMeans(n_clusters=n_struct, init='k-means++', n_init=100)
    return k_means.fit(obs)


def spectral_clustering(obs, n_struct):
    approx, _, _ = ssl.svds(obs, k=n_struct)
    return hierarchial_clustering(approx, n_struct)


def pecok_clustering(obs, n_struct, rho=5):
    gram_corrected = (obs.T.dot(obs) - np.diag(gamma.gamma_hat4(obs.T))) / obs.shape[0]
    U, _, V = ssl.svds(gram_corrected, k=n_struct)
    Bhat = admm.pecok_admm(gram_corrected, K=n_struct, rho=rho, mat_init=U.dot(V))
    return hierarchial_clustering(Bhat, n_struct=n_struct)


seed = 432
np.random.seed(seed)
print("seed is %i" % seed)

n_var = 10
n_obs = 100

print("\nVAR CLUSTERING\n\n")

truth = np.asmatrix(np.concatenate((np.repeat(0, n_var//2), np.repeat(1, n_var//2))))
membership = truth.T.dot(np.matrix([1, 0])) + (1-truth).T.dot(np.matrix([0, 1]))
stds = np.ones(n_var)
stds[:(n_var//2)] = 0.1
sigma = membership.dot(0.1*np.identity(2)).dot(membership.T) + np.diag(stds)
mat_data = np.random.multivariate_normal(mean=np.zeros(n_var), cov=sigma, size=n_obs)
gram_data = mat_data.T.dot(mat_data) / mat_data.shape[0]

print("truth:".ljust(15), truth)
print("hierarchical:".ljust(15), hierarchial_clustering(mat_data.T, n_struct=2).labels_)
print("kmeans:".ljust(15), kmeans_clustering(mat_data.T, n_struct=2).labels_)
print("spectral:".ljust(15), spectral_clustering(gram_data, n_struct=2).labels_)
print("pecok:".ljust(15), pecok_clustering(mat_data, n_struct=2).labels_)

print("\nPOINT CLUSTERING\n\n")

n_var = 100
n_obs = 10

truth = np.asmatrix(np.concatenate((np.repeat(0, n_obs//2), np.repeat(1, n_obs//2))))
X = np.zeros((n_obs, n_var))
snr = 0.3
X[:n_obs//2, :] = np.ones(n_var)*snr + np.random.normal(scale=1, size=(n_obs//2, n_var))
X[n_obs//2:, :] = -np.ones(n_var)*snr + np.random.normal(scale=0.1, size=(n_obs//2, n_var))
gram = X.dot(X.T) / X.shape[1]

print("truth:".ljust(15), truth)
print("hierarchical:".ljust(15), hierarchial_clustering(X, n_struct=2).labels_)
print("kmeans:".ljust(15), kmeans_clustering(X, n_struct=2).labels_)
print("spectral:".ljust(15), spectral_clustering(gram, n_struct=2).labels_)
print("pecok:".ljust(15), pecok_clustering(X.T, n_struct=2).labels_)
