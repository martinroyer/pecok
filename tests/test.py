"""Test for variable and point clustering"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT


import numpy as np
from pecok import pecok_clustering
from sklearn.cluster import KMeans


seed = 432
np.random.seed(seed)
print("seed is %i" % seed)

print("\nVAR CLUSTERING\n\n")
n_var = 10
n_obs = 100

truth = np.asmatrix(np.concatenate((np.repeat(0, n_var//2), np.repeat(1, n_var//2))))
membership = truth.T.dot(np.matrix([1, 0])) + (1-truth).T.dot(np.matrix([0, 1]))
stds = np.ones(n_var)
stds[:(n_var//2)] = 0.1
sigma = membership.dot(0.1*np.identity(2)).dot(membership.T) + np.diag(stds)
mat_data = np.random.multivariate_normal(mean=np.zeros(n_var), cov=sigma, size=n_obs)

print("truth:".ljust(15), truth)
print("kmeans:".ljust(15), KMeans(n_clusters=2, init='k-means++', n_init=100).fit(mat_data.T).labels_)
print("pecok:".ljust(15), pecok_clustering(mat_data.T, n_struct=2).labels_)
print("pecok:".ljust(15), pecok_clustering(mat_data.T, n_struct=2, int_corr=0).labels_)


print("\nPOINT CLUSTERING\n\n")
n_var = 100
n_obs = 10

truth = np.asmatrix(np.concatenate((np.repeat(0, n_obs//2), np.repeat(1, n_obs//2))))
X = np.zeros((n_obs, n_var))
snr = 0.3
X[:n_obs//2, :] = np.ones(n_var)*snr + np.random.normal(scale=1, size=(n_obs//2, n_var))
X[n_obs//2:, :] = -np.ones(n_var)*snr + np.random.normal(scale=0.1, size=(n_obs//2, n_var))

print("truth:".ljust(15), truth)
print("kmeans:".ljust(15), KMeans(n_clusters=2, init='k-means++', n_init=100).fit(X).labels_)
print("pecok:".ljust(15), pecok_clustering(X, n_struct=2, verbose=True).labels_)
print("pecok:".ljust(15), pecok_clustering(X, n_struct=2, rho=100, eps_residual=1e-6, n_iter_max=3000, verbose=True).labels_)
