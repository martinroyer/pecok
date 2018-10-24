"""Clustering wrapper"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.linalg import svds

from .gamma import gamma_hat
from .admm import pecok_admm


def corrected_relational(obs, int_corr):
    return (obs.dot(obs.T) - gamma_hat(obs, int_corr=int_corr)) / obs.shape[1]


def pecok_clustering(obs, n_struct, int_corr=4, **kwargs):
    gram_corrected = corrected_relational(obs, int_corr=int_corr)
    U, _, V = svds(gram_corrected, k=n_struct)
    Bhat = pecok_admm(gram_corrected, n_struct=n_struct, mat_init=None, **kwargs)
    return AgglomerativeClustering(linkage='ward', n_clusters=n_struct).fit(Bhat)
