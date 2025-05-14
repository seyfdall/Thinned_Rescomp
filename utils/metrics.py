import numpy as np
from math import comb
from scipy.stats import pearsonr

"""
Metrics
"""

def nrmse(true, pred):
    """ Normalized root mean square error. (A metric for measuring difference in orbits)
    Parameters:
        Two mxn arrays. Axis zero is assumed to be the time axis (i.e. there are m time steps)
    Returns:
        err (ndarray): Error at each time value. 1D array with m entries
    """
    # sig = np.std(true, axis=0)
    # err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
    err = np.linalg.norm((true-pred), axis=1, ord=2) # Just regular 2-norm
    return err


def valid_prediction_index(err, tol):
    """First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."""
    mask = np.logical_or(err > tol, ~np.isfinite(err))
    if np.any(mask):
        return np.argmax(mask)
    return len(err)


def vpt_time(ts, Uts, pre, vpt_tol=5.):
    """
    Valid prediction time for a specific instance.
    """
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, vpt_tol)
    if idx == 0:
        vptime = 0.
    else:
        vptime = ts[idx-1] - ts[0]
    return vptime


def div_metric_tests(states):
    """
    Compute diversity scores of internal reservoir states and their time derivatives,
    matching the element-wise absolute difference method from div_metric_tests.

    Parameters:
        states (ndarray): Reservoir states, shape (T, n)

    Returns:
        (float, float): Position-based and derivative-based diversity scores
    """
    T, n = states.shape
    res_deriv = np.gradient(states, axis=0)

    # Compute abs values
    abs_states = np.abs(states)
    abs_deriv = np.abs(res_deriv)

    # Compute pairwise differences with vectors
    diffs_pos = abs_states[:, :, None] - abs_states[:, None, :]
    diffs_der = abs_deriv[:, :, None] - abs_deriv[:, None, :]

    # Take upper triangle without diagonal to avoid double counting and self pairs
    triu_indices = np.triu_indices(n, k=1)

    # Sum over time, average over number of pairs
    div_pos = np.sum(np.abs(diffs_pos[:, triu_indices[0], triu_indices[1]])) / (T * comb(n, 2))
    div_der = np.sum(np.abs(diffs_der[:, triu_indices[0], triu_indices[1]])) / (T * comb(n, 2))

    return div_pos, div_der


def consistency_analysis_sphering(x, y, max_cutoff=8000, alpha=1e-9):
    """ Based on the Appendix: Consistency Analysis walkthrough sent by Dr. Lymburn
        Parameters:
        ----------
        x: ndarray(N,L)
            States using initial state r0. L = time, N = number of nodes

        y: ndarray(N,L)
            States using different initial state r0 prime. L = time, N = number of nodes

        Returns:
        --------
        cap: float
            The consistency capacity of the system
        S: ndarray(N)
            Each node's consistency
    """

    def _transient_cutoff(u,v,eps=1e-3):
        """Calculate first potential cutoff for transients"""
        # Look at FFT High-Frequency Transients and Variance Analysis
        err = np.linalg.norm(u-v, axis=0)
        cutoffs = np.where(err < eps)[0]
        transient_cutoff = max_cutoff
        if len(cutoffs) > 0:
            transient_cutoff = min(cutoffs[0], max_cutoff)
        return transient_cutoff
    
    transient_cutoff = _transient_cutoff(x,y)
    
    # Cutoff transient states
    x = x[:,transient_cutoff:]
    y = y[:,transient_cutoff:]

    N, L = np.shape(x)

    # The next lines are the Consistency PDF attempt to get unit variance

    # Calculate Covariance Matrices
    # Transpose here to ensure N x N Covariance matrix
    Cxx = x @ x.T / L 
    Cyy = y @ y.T / L 

    # Save space by averaging the two
    C = (Cxx + Cyy) / 2.

    # Add regularization term 
    C = C + alpha * np.eye(N)

    # Compute SVD
    U, S, _ = np.linalg.svd(C)
    Qxx = U
    S_inv = np.diag(1. / np.sqrt(S))

    # Apply spherical transformation T_o
    # Multiplying by S_inv in the spherical transformation is equivalent 
    # to dividing by standard deviation (conjecture)
    T_o = Qxx @ S_inv @ Qxx.T
    x = T_o @ x
    y = T_o @ y

    # Calculate Cross-variance matrix
    Cxy = x @ y.T / L
    Css = (Cxy + Cxy.T) / 2.

    # Calculuate Consistency Capacity
    cap = np.abs(np.trace(Css)) / N

    return cap


def consistency_analysis_pearson(x, y, transient_cutoff=3000, alpha=1e-9):
    """ Based on the Echo State Paper - Pearson Correlation Coefficient Definition
        Parameters:
        ----------
        x: ndarray(N,L)
            States using initial state r0. L = time, N = number of nodes

        y: ndarray(N,L)
            States using different initial state r0 prime. L = time, N = number of nodes

        Returns:
        --------
        cap: float
            The consistency capacity of the system
        S: ndarray(N)
            Each node's consistency
    """

    def _rescale_states(v):
        """Rescale states to have 0 mean and unit variance"""
        mean = np.mean(v, axis=1)
        std = np.std(v, axis=1)
        v_mod = v.T - mean
        valid = std > alpha
        v_mod[:,valid] = v_mod[:,valid] / std[valid]
        return v_mod.T
    
    # Cutoff transient states
    x = x[:,transient_cutoff:]
    y = y[:,transient_cutoff:]

    N, L = np.shape(x)

    # Rescale x and y
    x = _rescale_states(x)
    y = _rescale_states(y)

    cap = np.sum(x*y) / (N*L)

    return cap