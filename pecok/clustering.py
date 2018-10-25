"""Clustering wrapper"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.linalg import svd as LINsvd
from scipy.sparse.linalg import svds as SPAsvd

from .gamma import gamma_hat
from .admm import pecok_admm


def corrected_relational(obs, int_corr):
    return (obs.dot(obs.T) - gamma_hat(obs, int_corr=int_corr)) / obs.shape[1]


def pecok_clustering(obs, n_struct, int_corr=4, **kwargs):
    gram_corrected = corrected_relational(obs, int_corr=int_corr)
    U, _, V = SPAsvd(gram_corrected, k=n_struct)
    Bhat = pecok_admm(gram_corrected, n_struct=n_struct, mat_init=U.dot(V), **kwargs)
    return AgglomerativeClustering(linkage='ward', n_clusters=n_struct).fit(Bhat).labels_


def kmeanz_clustering(obs, n_struct, int_corr=4):
    gram_corrected = corrected_relational(obs, int_corr=int_corr)
    U, s, _ = LINsvd(gram_corrected)
    approx = U.dot(np.diag(np.sqrt(s)))
    return AgglomerativeClustering(linkage='ward', n_clusters=n_struct).fit(approx).labels_
