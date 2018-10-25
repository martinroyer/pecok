"""Test for variable and point clustering"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

from functools import wraps
import timeit

# import pyper
import numpy as np
from pecok import pecok_clustering, kmeanz_clustering
from sklearn.cluster import KMeans


def timethis(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = timeit.default_timer()
        result = f(*args, **kw)
        te = timeit.default_timer()
        return result, te-ts
    return wrap


def hdclassif(obs, n_struct):
    n,_ = obs.shape
    myR = pyper.R()
    myR.run('library(HDclassif)')
    myR.assign('obs', obs)
    myR.assign('n_struct', n_struct)
    try:
        myR.run('res <- hddc(obs, K=n_struct, model=c(1,2,7,9))')
        result = np.array(myR['res$class'])
    except:
        print("fail hdclassif")
        result = np.zeros(n)
    return result

seed = 432
np.random.seed(seed)
print("seed is %i" % seed)

methods = [
    [lambda X, K: KMeans(n_clusters=K, init='k-means++', n_init=100).fit(X).labels_, 'kmeans++'],
#    [hdclassif, 'HDDC'],
    [pecok_clustering, 'pecok'],
    [kmeanz_clustering, 'kmeanz']
]

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
for method, method_name in methods:
    job_result, job_time = timethis(method)(mat_data.T, 2)
    print(method_name.ljust(15), job_result)
    print("job_time: %.2f (s)".ljust(15) % job_time)


print("\nPOINT CLUSTERING\n\n")
n_var = 100
n_obs = 10

truth = np.asmatrix(np.concatenate((np.repeat(0, n_obs//2), np.repeat(1, n_obs//2))))
X = np.zeros((n_obs, n_var))
snr = 0.3
X[:n_obs//2, :] = np.ones(n_var)*snr + np.random.normal(scale=1, size=(n_obs//2, n_var))
X[n_obs//2:, :] = -np.ones(n_var)*snr + np.random.normal(scale=0.1, size=(n_obs//2, n_var))

print("truth:".ljust(15), truth)
for method, method_name in methods:
    job_result, job_time = timethis(method)(X, 2)
    print(method_name.ljust(15), job_result)
    print("job_time: %.2f (s)".ljust(15) % job_time)
# print("pecok:".ljust(15), pecok_clustering(X, n_struct=2, rho=100, n_iter_max=3000, verbose=True).labels_)
