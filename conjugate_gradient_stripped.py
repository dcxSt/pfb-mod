"""
Script containing some of the functionality from conjugate_gradient.py
but without mpl import and possibly optimized. I think mpl import was 
causing problems when running on Niagara.
"""

import numpy as np
import matrix_operators as m # A, AT, A_inv, A_inv_wiener, A_quantize
import pfb


def get_saved_idxs(n_per_save=5, prct=0.03, k=80, lblock=2048):
    """Returns prct% of indices of the most troublesome channels.

    Parameters
    ----------
    n_per_save : int
        On the channels with salvaged data, we salvage one data
        point out of every n_per_save data points.
    prct : float
        Percent of data points we can save. Float between 0 and 1.
    k : int
        Number of blocks in our data.
    lblock : int
        Number of samples per block.

    Returns
    -------
    int
        The computed width, i.e. the number of central channels
        for which we are salvaging data points.
    np.ndarray
        An array with the indices of all the data-points we are
        to save.
    """

    # `width` is the num of channels on which we save some data on.
    width = int(lblock * prct * n_per_save) # prct=0.03 is 3%
    # Relative indices of saved data points within single block
    saved_idxs_relative = np.arange(lblock//2 - width//2, lblock//2 + width//2)
    # We need n_per_save different sets of indices.
    # We package them in a list here:
    saved_idxs_pieces = [saved_idxs_relative[(i*width)//n_per_save:((i+1)*width)//n_per_save] for i in range(n_per_save)]
    saved_idxs = np.concatenate([
        np.concatenate(
            [piece + s*lblock + i*n_per_save*lblock for i in range((k + n_per_save - s - 1)//n_per_save)]
        ) for (s,piece) in enumerate(saved_idxs_pieces)
    ])
    return width, saved_idxs



def conjugate_gradient_descent(B, u, x0=None, rmin=0.1, max_iter=20, 
        k=80, lblock=2048, x_true=None, 
        title="RMSE smoothed gradient steps"): 
    """Minimize chi-squared by casting as conjugate gradient. 
    
    This conjugate gradient descent method approximately solves the 
    linear equasion Bx = u. B is a symmetric, positive definite matrix
    operator and u is the target. 
    
    Parameters
    ----------
    B : callable
        A function (which is actually a square matrix in disguise.) 
    u : np.ndarray
        Data vector. The rhs of the quadratic equasion we are solving. 
    x0 : np.ndarray
        Initial guess for x. 
    rmin : float
        The threshold value for stopping the descent. If the RMSE goes
        below this value, we're gucci. 
    max_iter : int
        The maximal number of descent iterations to make before stopping. 
    k : int
        Number of frames
    x_true : np.ndarray, optional
        The actual array, before adding any noise. This param is only 
        necessary if verbose is set to True. 
    title : str, optional
        The title of the plot. 
    
    Returns
    -------
    np.ndarray
        The noise-corrected array. 
    """
    # u is the data, B is the symmetric matrix operator (written as func)
    if type(x0) != np.ndarray: x0 = np.zeros(len(u)) # If x0 is None basically
    # Solve for x : Bx = u
    x = x0.copy()
    r = u - B(x0)
    if np.sqrt(np.dot(r,r)) < rmin: 
        return x
    p = r.copy()
    # Optionally, plot figure
    # Conj Gradient descent, iterate through max number of steps
    for i in range(max_iter): 
        # If it passes below the threashold RMSE, break the loop
        if np.sqrt(np.dot(r,r)) < rmin: 
            print(f"INFO: RMSE passed below threashold rmin={rmin}.\
                    Terminating CG descent.")
            break

        # Compute the action of B on p
        Bp = B(p)
        alpha = np.dot(r,r) / np.dot(p,Bp)
        x = x + alpha * p
        r_new = r - alpha * Bp
        beta = np.dot(r_new,r_new) / np.dot(r,r)
        p = r_new + beta * p
        r = r_new
    return x


def conj_grad_with_prior(
        x:np.ndarray,
        frac_prior:float=0.05,
        delta:float=0.5,
        k:int=80,
        lblock:int=2048,
        wiener_thresh:float=0.1,
        npersave:int=4,
        x0_wiener:np.ndarray=None
        ):
    """
    Computes PFB, quantizes, applies correction given frac_prior of saved
    input data.

    Parameters
    ----------
    x : np.ndarray
        Time-series signal
    frac_prior : float
        A number between 0 and 1 that represents the percentage of TS-samples in prior
    delta : float
        Width of the quantization interval, normalized by STD
    k : int
        Number of frames to process at a time
    lblock : int
        Length of a frame
    wiener_thresh : float
        The threshold for Wiener filtering, if zero, Wiener is not applied. 
    npersave : int 
        If we save y percent of indices, the optimal indices to save are where 
        the noise is most amplified, i.e. in the middle of each of the time-domain
        frames. However, we would be foolish to just pick the most ill conditioned
        channels, over-correcting the central channels and under-correcting the ones
        next to them. So we make a wider selection in the center of the frames, wider 
        by a factor of npersave, but we only pick 1/npersave of the channels in that 
        range on each frame. 
    x0_wiener : ndarray
        This is the Wiener filtered starting point on CG. Defaults to None. If none
        will compute the wienered iPFB.

    Return
    ------
    x0_wiener : np.ndarray
        Wiener filtered iPFB
    x_out : np.ndarray
        CG optimized iPFB
    """
    d = m.A_quantize(x, delta) # 4+4 bit quantized output of PFB
    N_inv = np.ones(len(x)) * (6/delta**2) # Diagonal noise matrix
    _,saved_idxs = get_saved_idxs(npersave, 0.1, k, lblock) # Get prior, i.e. saved TD samples
    delta_prior = 1/15 # hard code magic delta prior based on 8-bit ADC intuition
    delta_prior_normalized = x.std() * delta_prior
    prior = np.zeros(len(x)) # What is this?
    prior[saved_idxs] = pfb.quantize_real(x[saved_idxs].copy(), delta_prior_normalized)
    Q_inv = np.ones(len(x)) # Inverse noise matrix for prior, set to zeros for no prior on vals without additional information
    Q_inv[saved_idxs] = np.ones(len(saved_idxs)) * (12 / delta_prior**2)
    B = lambda ts: m.AT(N_inv * m.A(ts)) + Q_inv * ts # operator in simp mat eqn
    u = m.AT(N_inv * d) + Q_inv * prior
    """Optimize CHI-squared using conjugate gradient method"""
    x0 = None
    if x0_wiener is None:
        x0_wiener = np.real(m.A_inv_wiener(d, wiener_thresh)) 
    max_iter = min(18, int(2.6 + 100*frac_prior*2.5)) # Rule of thumb
    x_out = conjugate_gradient_descent(B, u, x0=x0_wiener, rmin=0.0,
            max_iter=max_iter, k=k, lblock=lblock, x_true=x,
            title="RMSE smoothed gradient steps 10% data salvaged")
    return x0_wiener, x_out
























