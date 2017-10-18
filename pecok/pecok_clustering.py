"""Gamma estimation"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

from .gamma import Gamma_hat4
from .admm import pecok_admm

###############################################################################
def cluster(X, K):
    """Implementation of PECOK estimator of B*

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Training instances to cluster."""
    return pecok_admm(X.dot(X.T)-Gamma_hat4(X), K)

def cluster_sbm(A, K):
    """Implementation of PECOK estimator of B*

    Parameters
    ----------
    A : adjacency matrix for network, shape=(n_samples, n_samples)
        Training instances to cluster."""
    return pecok_admm(A.dot(A), K)
