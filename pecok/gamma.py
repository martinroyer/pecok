"""Gamma estimation"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

import itertools

import numpy as np


###############################################################################
def cross_diff(A, B):
    return A[:,None] - B[None,:]

def Gamma_hat3(X):
    """Gamma_hat3 estimator from PECOK supplement, in O(n_samples^3 * n_features)

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Training instances to cluster."""
    n_samples, n_features = X.shape
    X2 = X / (np.linalg.norm(X, axis=1, keepdims=True)+1e-8)
    Vab = np.zeros((n_samples,n_samples))
    for a,b in itertools.combinations(range(n_samples),2):
        msk = [i for i in range(n_samples) if i != a and i != b]
        Vab[(a,b),(b,a)] = np.max(np.abs(X2[msk,:].dot((X[a,:]-X[b,:]))))
    Vab.flat[::n_samples+1] = np.inf
    gamma = np.asarray([(X[a,:]-X[np.argmin(Vab[a,:]),:]).dot(X[a,:]) for a in range(n_samples)])
    return gamma

def precompute_Vab_chunk(X, chunk_size):
    n_samples, n_features = X.shape
    preVab = np.zeros((n_samples**2,n_samples**2))

    chunk_size = np.min((chunk_size, n_samples ** 2))
    n_blocks = n_samples**2 // chunk_size
    groups = np.repeat(np.arange(n_blocks),chunk_size)
    rest = np.mod(n_samples**2,chunk_size)
    if rest > 0:
        groups = np.concatenate((groups, np.repeat(n_blocks, rest)))
    for chpa,chpb in itertools.combinations_with_replacement(range(len(np.unique(groups))),2):
        pcha = np.sum(groups == chpa)
        pchb = np.sum(groups == chpb)
        Xa = np.zeros((pcha,n_features),order='F')
        for idxc,c in enumerate(np.arange(n_samples**2)[groups == chpa]):
            Xa[idxc,:] = X[c//n_samples,:] - X[np.mod(c,n_samples),:]
        Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-8)
        Xb = np.zeros((pchb,n_features),order='F')
        for idxc,c in enumerate(np.arange(n_samples**2)[groups == chpb]):
            Xb[idxc,:] = X[c//n_samples,:] - X[np.mod(c,n_samples),:]
        Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-8)
        dotXab = np.abs(Xa.dot(Xb.T))
        preVab[chpa*chunk_size:chpa*chunk_size+pcha,chpb*chunk_size:chpb*chunk_size+pchb] = dotXab
        preVab[chpb*chunk_size:chpb*chunk_size+pchb,chpa*chunk_size:chpa*chunk_size+pcha] = dotXab.T
    return preVab

def precompute_Vab(X):
    n_samples, n_features = X.shape
    Xab = cross_diff(X, X).reshape((n_samples**2,n_features), order='F')
    Xab = Xab / (np.linalg.norm(Xab, axis=1, keepdims=True)+1e-8)
    return np.abs(Xab.dot(Xab.T))

def compute_Vab(X):
    n_samples, n_features = X.shape
    preVab = precompute_Vab(X) if n_features * n_samples ** 2 < 1e8\
        else precompute_Vab_chunk(X, chunk_size=np.max((1, np.int(2 * 1e8 // n_features))))
    Vab = np.zeros((n_samples,n_samples))
    for a, b in itertools.combinations(range(n_samples), 2):
        msk = [i + j * n_samples for i, j in itertools.combinations(
                                [i for i in range(n_samples) if i != a and i != b], 2)]
        Vab[(a,b),(b,a)] = np.max(preVab[b + a * n_samples, msk])
    Vab.flat[::n_samples + 1] = np.inf
    return Vab

def Gamma_hat4(X):
    """Gamma_hat4 estimator from PECOK, in O(n_samples^4 * n_features)

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Training instances to cluster."""
    n_samples, n_features = X.shape
    Vab = compute_Vab(X)
    gamma = np.zeros(n_samples)
    neighbours = [np.argpartition(Vab[a,:], 2)[0:2] for a in range(n_samples)]
    for a in range(n_samples):
        b1, b2 = neighbours[a]
        gamma[a] = (X[a,:] - X[b1,:]).dot(X[a,:] - X[b2,:])
    return np.asarray(gamma)
