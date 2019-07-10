"""PECOK clustering"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

import numpy as np
from scipy.linalg import svd as lin_svd
from scipy.sparse.linalg import svds as spa_svd

from sklearn.base          import BaseEstimator,  ClusterMixin, TransformerMixin
from sklearn.cluster       import AgglomerativeClustering

from .gamma import gamma_hat
from .admm import pecok_admm


def _corrected_relational(obs, corr):
    return (obs.dot(obs.T) - gamma_hat(obs, corr=corr)) / obs.shape[1]


def _kmeanz(X, n_clusters, corr):
    gram_corrected = _corrected_relational(X, corr=corr)
    U, s, _ = lin_svd(gram_corrected, compute_uv=True)
    approx = U.dot(np.diag(np.sqrt(s)))
    return approx, AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(approx)


def _pecok_clustering(obs, n_clusters, corr=4, **kwargs):
    gram_corrected = _corrected_relational(obs, corr=corr)
    U, _, V = spa_svd(gram_corrected, k=n_clusters)
    Bhat = pecok_admm(gram_corrected, n_clusters=n_clusters, mat_init=U.dot(V), **kwargs)
    return AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(Bhat)


class Pecok(BaseEstimator, ClusterMixin, TransformerMixin):
    """PeCoK clustering
        Read more in [my thesis: http://www.theses.fr/2018SACLS442]
        Parameters
        ----------
        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.
        corr : int, optional, default: 0
            Method for correcting for bias, defaults to 0:

            0: no correction
            clustering in a smart way to speed up convergence. See section
            Notes in k_init for more details.
            1: bad correction
            4: good correction
        init : @TODO{'k-means++', 'random' or an ndarray}
            Method for initialization, defaults to 'k-means++':

            'k-means++' : selects initial cluster centers for k-mean
            clustering in a smart way to speed up convergence. See section
            Notes in k_init for more details.

         Attributes
        ----------
        labels_ :
            Labels of each point

        Examples
        --------
        >>> from pecok import Pecok
        >>> import numpy as np
        >>> X = np.array([[1, 2], [1, 4], [1, 0],
        ...               [4, 2], [4, 4], [4, 0]])
        >>> pecok = Pecok(n_clusters=2).fit(X)
        >>> pecok.labels_
        array([1, 1, 1, 0, 0, 0], dtype=int32)
    """

    def __init__(self, n_clusters=8, corr=0, init='notyet',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):

        self.n_clusters = n_clusters
        self.corr = corr
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        """

        hc_ = \
            _pecok_clustering(
                X, n_clusters=self.n_clusters, corr=self.corr, verbose=self.verbose)
                # init=self.init, n_init=self.n_init,
                # max_iter=self.max_iter, verbose=self.verbose,
                # precompute_distances=self.precompute_distances,
                # tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                # n_jobs=self.n_jobs, algorithm=self.algorithm,
                # return_n_iter=True)
        self.labels_ = hc_.labels_
        return self


class KMeanz(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-MeanZ clustering
        Read more in [my thesis: http://www.theses.fr/2018SACLS442]
        Parameters
        ----------
        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.
        corr : int, optional, default: 0
            Method for correcting for bias, defaults to 0:

            0: no correction
            clustering in a smart way to speed up convergence. See section
            Notes in k_init for more details.
            1: bad correction
            4: good correction
        init : @TODO{'k-means++', 'random' or an ndarray}
            Method for initialization, defaults to 'k-means++':

            'k-means++' : selects initial cluster centers for k-mean
            clustering in a smart way to speed up convergence. See section
            Notes in k_init for more details.

         Attributes
        ----------
        corrected_points_: array, [n_points, n_features]
            Points representation in Z-space
        labels_ :
            Labels of each point

        Examples
        --------
        >>> from pecok import KMeanz
        >>> import numpy as np
        >>> X = np.array([[1, 2], [1, 4], [1, 0],
        ...               [4, 2], [4, 4], [4, 0]])
        >>> kmeanz = KMeanz(n_clusters=2).fit(X)
        >>> kmeanz.labels_
        array([1, 1, 1, 0, 0, 0], dtype=int32)
    """

    def __init__(self, n_clusters=8, corr=0, init='notyet',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):

        self.n_clusters = n_clusters
        self.corr = corr
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        """

        self.corrected_points_, hc_ = \
            _kmeanz(
                X, n_clusters=self.n_clusters, corr=self.corr)
                # init=self.init, n_init=self.n_init,
                # max_iter=self.max_iter, verbose=self.verbose,
                # precompute_distances=self.precompute_distances,
                # tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                # n_jobs=self.n_jobs, algorithm=self.algorithm,
                # return_n_iter=True)
        self.labels_ = hc_.labels_
        return self
