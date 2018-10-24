"""Gamma estimation"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

import itertools
import numpy as np



def gamma_hat1(X):
    gram_X = X.dot(X.T)
    diag_mem = np.copy(np.diag(gram_X))
    np.fill_diagonal(gram_X,-np.inf)
    gamma = diag_mem - np.max(gram_X,1)
    return gamma


def gamma_hat2(X):
    n_samples,_ = X.shape
    X2 = X / (np.linalg.norm(X, axis=1, keepdims=True)+1e-8)
    XaX2 = X.dot(X2.T)
    np.fill_diagonal(XaX2,-np.inf)
    gamma = np.zeros(n_samples)
    neighbours = [np.argpartition(-XaX2[a,:], 1)[0:1] for a in range(n_samples)]
    for a in range(n_samples):
        b1 = neighbours[a]
        gamma[a] = (X[a,:]-X[b1,:]).dot(X[a,:])
    return gamma


def gamma_hat3(X):
    """Gamma_hat3 estimator from PECOK supplement, in O(n_samples^3 * n_features)

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Training instances to cluster."""
    n_samples,_ = X.shape
    X2 = X / (np.linalg.norm(X, axis=1, keepdims=True)+1e-8)
    XaX2 = X.dot(X2.T)
    Vab = np.zeros((n_samples,n_samples))
    for a,b in itertools.combinations(range(n_samples),2):
        msk = [i for i in range(n_samples) if i != a and i != b]
        Vab[(a,b),(b,a)] = np.max(np.abs(XaX2[a,msk]-XaX2[b,msk]))
    Vab.flat[::n_samples+1] = np.inf
    gamma = np.asarray([(X[a,:]-X[np.argmin(Vab[a,:]),:]).dot(X[a,:]) for a in range(n_samples)])
    return gamma


def gamma_hat4(X):
    """Gamma_hat4 estimator from PECOK, in O(n_samples^4 * n_features)

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Training instances to cluster."""
    n_samples,_ = X.shape
    XaXb = X.dot(X.T)

    Wab = np.zeros((n_samples,n_samples**2))
    for c in range(n_samples):
        Wab[:,c*n_samples:(c+1)*n_samples] = XaXb[:,c]-XaXb
    Wab = Wab / (np.linalg.norm(Wab, axis=0, keepdims=True)+1e-8)
    Vab = np.zeros((n_samples,n_samples))
    for a, b in itertools.combinations(range(n_samples), 2):
        msk = [i + j * n_samples for i, j in itertools.combinations(
            [i for i in range(n_samples) if i != a and i != b], 2)]
        Vab[(a,b),(b,a)] = np.max(np.abs(Wab[a,msk]-Wab[b,msk]))
    Vab.flat[::n_samples + 1] = np.inf
    gamma = np.zeros(n_samples)
    neighbours = [np.argpartition(Vab[a,:], 2)[0:2] for a in range(n_samples)]
    for a in range(n_samples):
        b1, b2 = neighbours[a]
        gamma[a] = (X[a,:] - X[b1,:]).dot(X[a,:] - X[b2,:])
    return np.asarray(gamma)


def no_correction(X):
    return np.zeros(X.shape[0])


def gamma_hat(X, int_corr):
    ghat = {
        0: no_correction,
        1: gamma_hat1,
        2: gamma_hat2,
        3: gamma_hat3,
        4: gamma_hat4,
    }.get(int_corr, no_correction)(X)
    return np.diag(ghat)
