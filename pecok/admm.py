"""ADMM implementation for PECOK"""

# author: Martin Royer <martin.royer@math.u-psud.fr>
# License: MIT

import numpy as np
from scipy import linalg as la


def operator_lstarllstarinv_sym(u, v):
    """Operator \widetildetilde{L}^*_{sym} on (u,v) in R^{p+1} -> R^{p*p}"""
    temp = u.repeat(u.size).reshape((u.size, u.size))
    return (temp + temp.T)/2 + np.diag(np.repeat(v, u.size))


def proj_lin_Hsymmetric(Y, n_struct):
    """Projection onto \Pi_{\mathcal{A}sym}(Y)"""
    n_samples,_ = Y.shape
    x = np.sum(Y, 1) - 1
    y = np.trace(Y) - n_struct
    invx = (x-(np.sum(x)+y)/(2*n_samples))/(n_samples-1)
    invy = (y-(np.sum(x)+y)/(2*n_samples))/(n_samples-1)
    Y = Y - operator_lstarllstarinv_sym(invx, invy)
    return Y


def proj_positive(x, thresh=0):
    """Project onto component-positive matrix"""
    x[x < thresh] = 0
    return x


def proj_Snp_imp(Y):
    """Improved projection onto semi-definite positive matrix"""
    n_samples,_ = Y.shape
    eig_vals = la.eigh(Y, eigvals_only=True)
    n_val_neg = np.sum(eig_vals<0)
    if n_val_neg == 0:
        return Y
    if n_val_neg == n_samples:
        return np.zeros((n_samples,n_samples))
    if n_val_neg < n_samples-n_val_neg:
        eig_vals, v = la.eigh(-Y, eigvals=(n_samples - n_val_neg, n_samples - 1))
        Y = Y + v.dot(np.diag(eig_vals)).dot(v.T)
    else:
        eig_vals, v = la.eigh(Y, eigvals=(n_val_neg, n_samples - 1))
        Y = v.dot(np.diag(eig_vals)).dot(v.T)
    return Y


def pecok_admm(relational_data, n_struct, n_iter_max=-1, rho=5, mat_init=None, verbose=False, eps_residual=1e-4):
    """Implementation of Alternating Direction Method of Multipliers

    Parameters
    ----------
    relational_data : symmetric matrix of relational data (e.g. gram matrix), shape=(n_samples, n_samples)
        Training instances to cluster."""
    n_samples,_ = relational_data.shape
    if n_iter_max < 0:
        n_iter_max = np.max((1000,2*n_samples))
    relational_data = relational_data / np.linalg.norm(relational_data)

    X, Y, Z = np.identity(n_samples), np.identity(n_samples), np.identity(n_samples)
    if mat_init is not None:
        X, Y, Z = mat_init, mat_init, mat_init
    U, V, W = np.zeros((n_samples,n_samples)), np.zeros((n_samples,n_samples)), np.zeros((n_samples,n_samples))
    Xbar = (X + Y + Z)/3

    n_iter = 0
    while n_iter < n_iter_max:
        n_iter = n_iter + 1

        oldXbar = Xbar
        X = proj_lin_Hsymmetric(Xbar - U + relational_data / rho, n_struct)
        Y = proj_positive(Xbar - V)
        Z = proj_Snp_imp(Xbar - W)
        Xbar = (X + Y + Z)/3

        U = U + X - Xbar
        V = V + Y - Xbar
        W = W + Z - Xbar

        res_dual = rho * np.linalg.norm(Xbar-oldXbar)
        res_primal = np.linalg.norm((X-Xbar, Z-Xbar, Y-Xbar))
        if not (is_primal_high(eps_residual, res_primal, X, Y, Z) or is_dual_high(eps_residual, res_dual, Y, Z)):
            break
    if verbose:
        print("ADMM ends -- n_iter=%i, rho=%2.2f" % (n_iter, rho))
        print("          -- res_primal=%.3e, res_dual=%.3e" % (res_primal, res_dual))
    return Z


def is_primal_high(eps_residual, res_primal, X, Y, Z):
    return res_primal > eps_residual * np.max((np.linalg.norm(X), np.linalg.norm(Y), np.linalg.norm(Z)))


def is_dual_high(eps_residual, res_dual, Y, Z):
    return res_dual > eps_residual * (np.sqrt(Y.shape[0]) + np.linalg.norm(Y) + np.linalg.norm(Z))
