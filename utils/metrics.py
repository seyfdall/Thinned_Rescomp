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
    """ Compute Diversity scores of predictions
    """
    # Take the derivative of the pred_states
    res_deriv = np.gradient(states, axis=0)
    T, n = states.shape

    # Run the metric for the old and new diversity scores
    div_pos = 0
    div_der = 0
    for i in range(n-1):
        for j in range(i+1, n):
            div_pos += np.sum(np.abs(np.abs(np.abs(states[:T, i]) - np.abs(states[:T, j]))))
            div_der += np.sum(np.abs(np.abs(res_deriv[:T, i]) - np.abs(res_deriv[:T, j])))
    denom = T*comb(n,2)
    div_pos = div_pos / denom
    div_der = div_der / denom

    return div_pos, div_der


def pearson_consistency_metric(states, states_perturbed):
    """ Compute the consistency metric for predicted states based on the Echo State Paper - Pearson Correlation Coefficient 
        Parameters:
        ----------
        states: ndarray(T,n)
            States using unperturbed initial state r0. T = time, n = number of nodes

        states_perturbed: ndarray(T,n)
            Perturbed states using perturbed initial state r0. T = time, n = number of nodes

        Returns:
        --------
        aggregated_pearson_correlation_coeff: float 
            The aggregated pearson correlation coefficient between the two response states
    """
    if len(states.shape) == 1:
        return pearsonr(states, states_perturbed)[0]
    else:
        T, n = states.shape
        gammas = np.zeros(n)
        for i in range(n):
            gammas[i] = pearsonr(states[:,i], states_perturbed[:,i])[0]
        # pearsonr is not defined for constant state vectors so nan will be returned - remove these from the array
        aggregated_pearson_correlation_coeff = np.mean(gammas[np.isfinite(gammas)])
        return aggregated_pearson_correlation_coeff


def consistency_analysis(x, y, transient_cutoff=1000, alpha=1e-9):
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

    def _rescale_states(v):
        """Rescale states to have 0 mean and unit variance"""
        mean = np.mean(v, axis=0)
        std = np.std(v)
        return (v - mean) / std if np.abs(std) > alpha else v
    
    def _transient_cutoff(u,v,eps=1e-3):
        """Calculate first potential cutoff for transients"""
        err = np.linalg.norm(u-v, axis=0)
        cutoffs = np.where(err < eps)[0]
        if len(cutoffs) == 0:
            return -1
        return cutoffs[0]
    
    transient_cutoff = _transient_cutoff(x,y)

    # If no cutoff found, then replica states never converged
    if transient_cutoff == -1:
        return 0
    
    # Cutoff transient states
    x = x[:,transient_cutoff:]
    y = y[:,transient_cutoff:]

    N, L = np.shape(x)

    # Rescale x and y
    x = _rescale_states(x)
    y = _rescale_states(y)

    # Calculate Covariance Matrices
    # Transpose here to ensure N x N Covariance matrix
    Cxx = x @ x.T / L 
    Cyy = y @ y.T / L 

    # Save space by averaging the two
    C = (Cxx + Cyy) / 2.

    # Add regularization term 
    C = C + alpha * np.eye(N)

    # Compute SVD
    U, S, Vh = np.linalg.svd(C)
    Qxx = U
    S_inv = np.diag(1. / np.sqrt(S))

    # Apply spherical transformation T_o
    T_o = Qxx @ S_inv @ Qxx.T
    x = T_o @ x
    y = T_o @ y

    # Calculate Cross-variance matrix
    Cxy = x.T @ y / L
    Css = (Cxy + Cxy.T) / 2.

    # Calculuate Consistency Capacity
    cap = np.abs(np.trace(Css)) / N

    return cap