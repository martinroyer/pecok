"""Test for variable and point clustering"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

from functools import partial
import timeit

import numpy as np
from pecok import KMeanz, Pecok
from sklearn.cluster import KMeans, AgglomerativeClustering


seed = 432
np.random.seed(seed)
print("seed is %i" % seed)

methods = [
    [partial(KMeans, init="k-means++", n_init=100), "k-means++"],
    [partial(AgglomerativeClustering, linkage='ward'), "Hierarchical"],
    [partial(KMeanz, corr=2), "KMeanz"],
    [partial(Pecok, corr=2), "Pecok"],
]

print("\nVAR CLUSTERING\n\n")
n_var = 10
n_obs = 100

truth = np.asmatrix(np.concatenate((np.repeat(0, n_var//2), np.repeat(1, n_var//2))))
membership = truth.T.dot(np.array([1, 0])[np.newaxis,:]) + (1-truth).T.dot(np.array([0, 1])[np.newaxis,:])
stds = np.ones(n_var)
stds[:(n_var//2)] = 0.1
sigma = membership.dot(0.1*np.identity(2)).dot(membership.T) + np.diag(stds)
mat_data = np.random.multivariate_normal(mean=np.zeros(n_var), cov=sigma, size=n_obs)

print("truth:".ljust(15), truth)
for method, method_name in methods:
    ts = timeit.default_timer()
    job_result = method(n_clusters=2).fit(mat_data.T).labels_
    te = timeit.default_timer()
    print(method_name.ljust(15), job_result)
    print("job_time: %.2f (s)".ljust(15) % (te-ts))


print("\nPOINT CLUSTERING\n\n")
n_obs = 10
n_var = 100

truth = np.asmatrix(np.concatenate((np.repeat(0, n_obs//2), np.repeat(1, n_obs//2))))
X = np.zeros((n_obs, n_var))
snr = 0.3
X[:n_obs//2, :] = np.ones(n_var)*snr + np.random.normal(scale=1, size=(n_obs//2, n_var))
X[n_obs//2:, :] = -np.ones(n_var)*snr + np.random.normal(scale=0.1, size=(n_obs//2, n_var))

print("truth:".ljust(15), truth)
for method, method_name in methods:
    ts = timeit.default_timer()
    job_result = method(n_clusters=2).fit(X).labels_
    te = timeit.default_timer()
    print(method_name.ljust(15), job_result)
    print("job_time: %.2f (s)".ljust(15) % (te-ts))
