import numpy as np
import pfb
from matrix_operators import A,AT, A_inv, A_inv_wiener, A_quantize
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from datetime import datetime

#%% Helper methods / routines

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

def get_uniform_saved_indices(prct=0.1, k=80, lblock=2048):
    """Returns prct% of indices uniformly from all channels

    prct : float
        The percentage of saved field samples. 

    k : int
        The number of frames. 

    lblock : int
        Frame length. ("length of block" as i used to call it)
    """
    nperframe = int(prct * lblock) # number that we can save per frame
    saved_idxs = np.zeros(nperframe * k,dtype="int")
    rel_idxs = np.arange(nperframe) # relative indexes saved in this frame
    for i in range(k):
        rel_idxs += nperframe
        rel_idxs %= lblock
        saved_idxs[i*nperframe + np.arange(nperframe)] = rel_idxs + i*lblock
    return saved_idxs


def mav(x, n=4):
    """Moving average. 

    This method is used for smoothing rugged simulated data. Largly in 
    order to make prettier plots. 
    """
    # Zero pad n copies of the array so that they allign
    y = [np.concatenate([np.zeros(i), x, np.zeros(n-i)]) for i in range(n)]
    # Average n neighbours
    y = np.mean(y, axis=0)
    # Trunkate and return array of same size
    return y[n//2:-n//2]


## TODO, work in progress
#def mcmc(B, u, x0=None, rmin=0.1, max_iter=20, 
#        k=80, lblock=2048, verbose=False, x_true=None): 
#    """Minimize chi-squared by casting as gradient descent problem. 
#    
#    This conjugate gradient descent method approximately solves the 
#    linear equasion Bx = u. B is a symmetric, positive definite matrix
#    operator and u is the target. 
#    
#    Parameters
#    ----------
#    B : callable
#        A function which is actually a square matrix in disguise. 
#    u : np.ndarray
#        Data vector. The rhs of the quadratic equasion we are solving. 
#    x0 : np.ndarray
#        Initial guess for x. 
#    rmin : float
#        The threshold value for stopping the descent. If the RMSE goes
#        below this value, we're gucci. 
#    max_iter : int
#        The maximal number of descent iterations to make before stopping. 
#    verbose : bool, optional
#        If verbose is set to True, it will plot the descent. 
#    x_true : np.ndarray, optional
#        The actual array, before adding any noise. This param is only 
#        necessary if verbose is set to True. 
#    title : str, optional
#        The title of the plot. 
#    saveas : str, optional
#        If a string is passed (& verbose is True), the figure will be
#        saved. 
#    
#    Returns
#    -------
#    np.ndarray
#        The noise-corrected array. 
#    """
#    # u is the data, B is the symmetric matrix operator (written as func)
#    if type(x0) != np.ndarray: x0 = np.zeros(len(u)) # If x0 is None basically
#    # Solve for x : Bx = u
#    x = x0.copy()
#    r = u - B(x0)
#    if np.sqrt(np.dot(r,r)) < rmin: 
#        return x
#    p = r.copy()
#    # Optionally, plot figure
#    if verbose is True: plt.figure(figsize=(14,4))
#    # Conj Gradient descent, iterate through max number of steps
#    for i in range(max_iter): 
#        # Opitonally, plot the residuals on each iteration
#        if verbose is True and (i%2==0 or i<4):
#            rms = (x_true - x)**2
#            rms = np.reshape(rms[5*lblock:-5*lblock], (k-10, lblock))
#            rms_net = np.sqrt(np.mean(rms)) # net (or total) rms
#            rms = np.sqrt(np.mean(rms, axis=0))
#            rms_smoothed = mav(rms, 20)[20:-20] # Chop off spoiled values
#            plt.plot(rms_smoothed, 
#                    label="step_n={} rms_net={:.4f}".format(i, rms_net),
#                    color=(i/(max_iter-1), (1.5*i/(max_iter-1)-0.75)**2, 1.0-i/(max_iter-1), 0.6)
#                    )
#
#        # If it passes below the threashold RMSE, break the loop
#        if np.sqrt(np.dot(r,r)) < rmin: 
#            print(f"INFO: RMSE passed below threashold rmin={rmin}.\
#                    Terminating CG descent.")
#            break
#
#        # Compute the action of B on p
#        Bp = B(p)
#        alpha = np.dot(r,r) / np.dot(p,Bp)
#        x = x + alpha * p
#        r_new = r - alpha * Bp
#        beta = np.dot(r_new,r_new) / np.dot(r,r)
#        p = r_new + beta * p
#        r = r_new
#    
#    # Conditionally plot config, annotations, etc
#    if verbose is True:
#        plt.legend()
#        plt.grid()
#        plt.title(title, fontsize=20)
#        plt.xlabel("Timestream Column Index",fontsize=16)
#        plt.ylabel("Normalized RMSE",fontsize=16)
#        plt.tight_layout()
#        if saveas is not None:
#            plt.savefig(saveas)
#        plt.show()
#    print("INFO: Conjugate Gradient descent completed.") 
#    return x


def get_Bu(x,d,saved_idxs,delta,stds=None,nchan=None):
    """"This helper routine returns the B matrix and u array that feed
    into conjugate_gradient_descent.

    Parameters
    ----------
    x : np.ndarray
        Input time-stream (raw digitized electric-field samples). 
        Represents truth.
    d : np.ndarray
        The PFB'd data. Flattened. 
    saved_idxs : np.ndarray
        An array of indices we salvage from the original time-stream.
    delta : float
        The quantization interval. (After PFB has been applied, in 
        real and imag)
    stds : np.ndarray
        An array of standard deviations of length nchan. This argument 
        can be supplied instead and treats the power on each channel
        seperately

    Returns
    -------
    callable
        The B matrix, who's type is a callable closure/function
    np.ndarray
        The u array, the target of the linear equation Bx=u we wish
        to solve
    """
    if stds is not None:
        assert nchan is not None, "Must supply nchan if stds are given"
        assert stds.shape == (nchan,), "Dimensions of stds array must be (nchan,)"
        #TODO CONTINUE HERE
        #nframes = len(x)//(nchan*2+1)
    # Ninv and Qinv are diagonal noise matrices (stored as 1d arrays)
    Ninv=np.ones(len(x)) * 6/delta**2
    prior=np.ones(len(x)) # prior information salvaged
    # ---BEGIN DEPRICATED---
    # The data we salvaged is also quantized
    prior[saved_idxs]=pfb.quantize_real(x[saved_idxs].copy(),delta)
    # ---END   DEPRICATED---
    # Q_inv inits to ones because we use the fact that our data is expected
    # to sample from a Gaussian Random Variable. Our prior on those samples
    # we don't salvage is zero +- 1, Our prior on samples we do salvage 
    # is that value +- delta/sqrt(12)
    Qinv=np.ones(len(x)) 
    Qinv[saved_idxs]=np.ones(len(saved_idxs))*(12/delta**2) 
    B=lambda ts:AT(Ninv * A(ts)) + Qinv*ts
    u=AT(Ninv*d) + Qinv*prior
    return B,u 

def conjugate_gradient_descent(B, u, x0=None, rmin=0.1, max_iter=20, 
        k=80, lblock=2048, verbose=False, x_true=None, 
        title="RMSE smoothed gradient steps",
        saveas=None): 
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
    verbose : bool, optional
        If verbose is set to True, it will plot the descent. 
    x_true : np.ndarray, optional
        The actual array, before adding any noise. This param is only 
        necessary if verbose is set to True. 
    title : str, optional
        The title of the plot. 
    saveas : str, optional
        If a string is passed (& verbose is True), the figure will be
        saved. 
    
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
    if verbose is True: plt.figure(figsize=(14,4))
    # Conj Gradient descent, iterate through max number of steps
    for i in range(max_iter): 
        # Opitonally, plot the residuals on each iteration
        if verbose is True and (i%2==0 or i<4):
            rms = (x_true - x)**2
            rms = np.reshape(rms[5*lblock:-5*lblock], (k-10, lblock))
            rms_net = np.sqrt(np.mean(rms)) # net (or total) rms
            rms = np.sqrt(np.mean(rms, axis=0))
            rms_smoothed = mav(rms, 20)[20:-20] # Chop off spoiled values
            plt.plot(rms_smoothed, 
                    label="step_n={} rms_net={:.4f}".format(i, rms_net),
                    color=(i/(max_iter-1), (1.5*i/(max_iter-1)-0.75)**2, 1.0-i/(max_iter-1), 0.6)
                    )

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
    
    # Conditionally plot config, annotations, etc
    if verbose is True:
        plt.legend()
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel("Timestream Column Index",fontsize=16)
        plt.ylabel("Normalized RMSE",fontsize=16)
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        plt.show()
    #print("INFO: Conjugate Gradient descent completed.") 
    return x


def conj_grad_with_prior(
        x:np.ndarray, 
        frac_prior:float=0.05,
        delta:float=0.5,
        k:int=80,
        lblock:int=2048, 
        verbose:bool=False,
        wiener_thresh:float=0.1,
        npersave:int=4, # For 4+4-bit quantization {1%:7,3%:5,5%:4,10%:3}
        x0_wiener=None
        ):
    """
    Computes PFB, quantizes, applies correction given 100 * frac_prior % of data.
    Plots the RMSE with vs without Wiener filter, then with and without correction.
    
    Parameters
    ----------
    x : np.ndarray
        Time-series signal
    frac_prior : float
        A number between 0 and 1 that represents the percentage of the prior given
    delta : float
        Width of the quantization interval
    k : int
        Number of frames to process at a time
    lblock : int
        Length of a frame
    verbose : bool
        If True, make and save plots. 
    wiener_thresh : float
        The threshold for Wiener filtering, if zero, no wiener filter is computed.
    npersave : int
        If we save y percent of indices, the optimal indices to save are where the
        noise is most amplified, i.e. in the middle of each time-domain frame. 
        However, we would be foolish to just pick the most ill conditioned channels, 
        over-correcting the central channels and under-correcting the ones next to
        them. So we make a wider selection in the center of the frames, wider by 
        a factor of npersave, but we only pick 1/npersave of the channels in that 
        range on each frame. 
    x0_wiener : ndarray
        This is the wiener filtered starting point of CG. Defaults to None. If none
        will compute the wienered iPFB. 

    Returns
    -------
    x0_wiener : np.ndarray
        Wiener filtered iPFB
    x_out : np.ndarray
        CG optimized iPFB
        
    """
    # Choose a photocopy & color-blind friendly colormap
    colors=plt.get_cmap('Set2').colors # get list of RGB color values
    # `d` is the output of the PFB with 4+4-bit quantization. 
    d = A_quantize(x,delta) 
    # `N_inv` and `Q_inv` are *diagonal* matrices, so we store them as 1D-arrays 
    N_inv = np.ones(len(x)) * 6 / delta**2 

    """100*frac_prior percent of original data given as prior."""
    # Get the indices for all the data points we 'salvage' in data collection
    _,saved_idxs = get_saved_idxs(npersave, 0.1, k, lblock)
    # The noise matrix for the prior. 
    prior = np.ones(len(x)) # What we know about x, information we salvaged. 
    # The data we save will also be 8-bit quantized. 
    prior[saved_idxs] = pfb.quantize_real(x[saved_idxs].copy() , delta) # Quantized original signal. 
    
    # Assumes prior on distribution, gaussian distributed with std=1
    Q_inv = np.ones(len(x)) # this is a prior, change to zeros if you want zero for infinite uncertainty
    Q_inv[saved_idxs] = np.ones(len(saved_idxs)) * (12 / delta**2) # 8 bits per real number (finer std because no complex) 
    
    B = lambda ts: AT(N_inv * A(ts)) + Q_inv * ts # think ts===x
    u = AT(N_inv * d) + Q_inv * prior # this is same as mult prior by var=12/delta^2
 
    """Optimize CHI squared using conjugate gradient method."""
    # x0 is the standard IPFB reconstruction
    x0 = None
    if x0_wiener is None:
        x0_wiener = np.real(A_inv_wiener(d, wiener_thresh)) # Weiner threshold 0.25
    
    # print("\n\nd={}".format(d)) # trace, they are indeed real
    
    # print("\nConjugate Gradient Descent, with 3% extra data (prior is a quantized 3% of original timestream)")
    
    if verbose:
        x0 = np.real(A_inv(d)) 
        # rms virgin pfb
        rms_virgin = (x - x0)**2
        rms_virgin = np.reshape(rms_virgin[5*lblock:-5*lblock],(k-10,lblock)) # bad practice to hard code k=80...? I just want to write this fast
        rms_net_virgin = np.sqrt(np.mean(rms_virgin))
        rms_virgin = np.sqrt(np.mean(rms_virgin,axis=0))
        rms_virgin = mav(rms_virgin,5)
        
        # rms wiener filtered pfb
        rms_wiener = (x - x0_wiener)**2
        rms_wiener = np.reshape(rms_wiener[5*lblock:-5*lblock],(k-10,lblock)) 
        rms_net_wiener = np.sqrt(np.mean(rms_wiener))
        rms_wiener = np.sqrt(np.mean(rms_wiener,axis=0))
        plt.figure(figsize=(14,4))
        plt.semilogy(rms_wiener[5:-5],label=f"rmse wiener filtered, rms_net={rms_net_wiener}",color=colors[1]) 
        plt.semilogy(rms_virgin[5:-5],label=f"rmse virgin ipfb, rms_net={rms_net_virgin}",color=colors[0]) 
        
        plt.grid(which="both") 
        plt.legend()
        #plt.title("Log IPFB RMS residuals (smoothed)\nrmse virgin = {:.3f} rmse wiener = {:.3f}".format(rms_net_virgin,rms_net_wiener),fontsize=20) 
        plt.title("IPFB Root Mean Squared residuals (smoothed)",fontsize=20)
        #plt.xlabel("Channel #",fontsize=13)
        plt.ylabel("RMSE",fontsize=16)
        plt.xlabel("Timestream Column Index",fontsize=16)
        plt.tight_layout()
        plt.savefig("img/RMSE_log_virgin_IPFB_residuals_wiener.png")
        plt.show(block=False)
        plt.pause(0.01)
    
    # Chose optimal max iter params
    if delta<=0.3: 
        max_iter10,max_iter5,max_iter3,max_iter1=3,2,2,2
    else: # usually it's 0.5
        max_iter10,max_iter5,max_iter3,max_iter1=18,15,10,5
    max_iter = min(18, int(2.6 + 100*frac_prior*2.5)) # Rule of thumb
    # RMS conj gradient descent
    saveas = f"img/RMSE_conjugate_gradient_descent_{100*frac_prior:.1f}percent.png" if verbose else None
    x_out = conjugate_gradient_descent(B, u, x0=x0_wiener, rmin=0.0,
            max_iter=max_iter, k=k, lblock=lblock, verbose=verbose, x_true=x, 
            title="RMSE smoothed gradient steps 10% data salvaged",
            saveas=saveas)
    return x0_wiener, x_out
 



#%% Plotting Routines
# This routine was written in the spirit of 
# "The good is the enemy of the done"...
# you have been warned
def conj_grad_one_three_five_perc(
        x:np.ndarray, 
        delta:float=0.5,
        k:int=80,
        lblock:int=2048, 
        verbose:bool=False,
        wiener_thresh:float=0.1,
        npersave1:int=7,
        npersave3:int=5,
        npersave5:int=4,
        npersave10:int=3
        ):
    """
    Computes PFB, quantizes, applies correction given 1%, 3%, 5% of data.
    Plots the RMSE with vs without Wiener filter, then with and without correction.
    
    Parameters
    ----------
    x : np.ndarray
        Time-series signal
    delta : float
        Width of the quantization interval
    k : int
        Number of frames to process at a time
    lblock : int
        Length of a frame

    Returns
    -------
    np.ndarray
        
    """
    # Choose a photocopy & color-blind friendly colormap
    colors=plt.get_cmap('Set2').colors # get list of RGB color values
    # `d` is the output of the PFB with 4+4-bit quantization. 
    d = A_quantize(x,delta) 
    # `N_inv` and `Q_inv` are *diagonal* matrices, so we store them as 1D-arrays 
    N_inv = np.ones(len(x)) * 6 / delta**2 


    get_indices_uniform=False # sample indices from specific channels
    if delta <= 0.1: get_indices_uniform=True # just sample uniformly


    """10 percent of original data given as prior."""
    # Get the indices for all the data points we 'salvage' in data collection
    if get_indices_uniform is True:
        saved_idxs_10 = get_uniform_saved_indices(0.1, k, lblock)
    else:
        _,saved_idxs_10 = get_saved_idxs(npersave10, 0.1, k, lblock)
    # The noise matrix for the prior. 
    prior_10 = np.ones(len(x)) # What we know about x, information we salvaged. 
    # The data we save will also be 8-bit quantized. 
    prior_10[saved_idxs_10] = pfb.quantize_real(x[saved_idxs_10].copy() , delta) # Quantized original signal. 
    
    # Assumes prior on distribution, gaussian distributed with std=1
    Q_inv_10 = np.ones(len(x)) # this is a prior, change to zeros if you want zero for infinite uncertainty
    Q_inv_10[saved_idxs_10] = np.ones(len(saved_idxs_10)) * (12 / delta**2) # 8 bits per real number (finer std because no complex) 
    
    B_10 = lambda ts: AT(N_inv * A(ts)) + Q_inv_10 * ts # think ts===x
    u_10 = AT(N_inv * d) + Q_inv_10 * prior_10 # this is same as mult prior by var=12/delta^2
 

    """5 percent of original data given as prior."""
    # Get the indices for all the data points we 'salvage' in data collection
    if get_indices_uniform is True:
        saved_idxs_5 = get_uniform_saved_indices(0.05, k, lblock)
    else:
        _,saved_idxs_5 = get_saved_idxs(npersave5, 0.05, k, lblock)
    # The noise matrix for the prior. 
    prior_5 = np.ones(len(x)) # What we know about x, information we salvaged. 
    # The data we save will also be 8-bit quantized. 
    prior_5[saved_idxs_5] = pfb.quantize_real(x[saved_idxs_5].copy() , delta) # Quantized original signal. 
    
    # Assumes prior on distribution, gaussian distributed with std=1
    Q_inv_5 = np.ones(len(x)) # this is a prior, change to zeros if you want zero for infinite uncertainty
    Q_inv_5[saved_idxs_5] = np.ones(len(saved_idxs_5)) * (12 / delta**2) # 8 bits per real number (finer std because no complex) 
    
    B_5 = lambda ts: AT(N_inv * A(ts)) + Q_inv_5 * ts # think ts===x
    u_5 = AT(N_inv * d) + Q_inv_5 * prior_5 # this is same as mult prior by var=12/delta^2
    
    
    """3 percent of original data given as prior."""
    # Get the indices for all data points we salvage
    if get_indices_uniform is True:
        saved_idxs_3 = get_uniform_saved_indices(0.03, k, lblock)
    else:
        _,saved_idxs_3 = get_saved_idxs(npersave3, 0.03, k, lblock)
    # The noise matrix for the prior. 
    prior_3 = np.zeros(len(x)) # What we know about x, information we salvaged. 
    # The data we save will also be 8-bit quantized. 
    prior_3[saved_idxs_3] = pfb.quantize_real(x[saved_idxs_3].copy() , delta) # Quantized original signal. 
    
    Q_inv_3 = np.ones(len(x)) # this is a prior, change to zeros if you want zero for infinite uncertainty
    Q_inv_3[saved_idxs_3] = np.ones(len(saved_idxs_3)) * (12 / delta**2) # 8 bits per real number (finer std because no complex) 
    
    B_3 = lambda ts: AT(N_inv * A(ts)) + Q_inv_3 * ts # think ts===x
    u_3 = AT(N_inv * d) + Q_inv_3 * prior_3 # this is same as mult prior by var=12/delta^2
    
    """1 percent of original data given as prior."""
    # Get the indices for data points we salvage
    if get_indices_uniform is True:
        saved_idxs_1 = get_uniform_saved_indices(0.01, k, lblock)
    else:
        _,saved_idxs_1 = get_saved_idxs(npersave1, 0.01, k, lblock)
    # the noise matrix for the prior
    prior_1 = np.zeros(len(x)) # what we know about x, information we saved
    prior_1[saved_idxs_1] = pfb.quantize_real(x[saved_idxs_1].copy() , delta) # quantized original signal
    
    # Q_inv inits to ones because we use the fact that our data is expected
    # to sample from a Gaussian Random Variable. Our prior on those samples
    # we don't salvage is zero +- 1, Our prior on samples we do salvage 
    # is that value +- delta/sqrt(12)
    Q_inv_1 = np.ones(len(x)) 
    Q_inv_1[saved_idxs_1] = np.ones(len(saved_idxs_1)) * 12 / delta**2 # 8 bits per real number
    
    B_1 = lambda ts: AT(N_inv * A(ts)) + Q_inv_1 * ts # think ts===x
    u_1 = AT(N_inv * d) + Q_inv_1 * prior_1
    
    """Optimize CHI squared using conjugate gradient method."""
    # x0 is the standard IPFB reconstruction
    x0 = np.real(A_inv(d)) 
    x0_wiener = np.real(A_inv_wiener(d, wiener_thresh)) # Weiner threshold 0.25
    
    # print("\n\nd={}".format(d)) # trace, they are indeed real
    # print("\n\nx_0={}".format(x0)) # complex dtype but zero imag componant
    
    # print("\nConjugate Gradient Descent, with 3% extra data (prior is a quantized 3% of original timestream)")
    
    # rms virgin pfb
    rms_virgin = (x - x0)**2
    rms_virgin = np.reshape(rms_virgin[5*lblock:-5*lblock],(k-10,lblock)) # bad practice to hard code k=80...? I just want to write this fast
    rms_net_virgin = np.sqrt(np.mean(rms_virgin))
    rms_virgin = np.sqrt(np.mean(rms_virgin,axis=0))
    rms_virgin = mav(rms_virgin,5)
    
    # rms wiener filtered pfb
    rms_wiener = (x - x0_wiener)**2
    rms_wiener = np.reshape(rms_wiener[5*lblock:-5*lblock],(k-10,lblock)) 
    rms_net_wiener = np.sqrt(np.mean(rms_wiener))
    rms_wiener = np.sqrt(np.mean(rms_wiener,axis=0))
    if verbose:
        plt.figure(figsize=(14,4))
        plt.semilogy(rms_wiener[5:-5],label=f"rmse wiener filtered, rms_net={rms_net_wiener}",color=colors[1]) 
        plt.semilogy(rms_virgin[5:-5],label=f"rmse virgin ipfb, rms_net={rms_net_virgin}",color=colors[0]) 
        
        plt.grid(which="both") 
        plt.legend()
        #plt.title("Log IPFB RMS residuals (smoothed)\nrmse virgin = {:.3f} rmse wiener = {:.3f}".format(rms_net_virgin,rms_net_wiener),fontsize=20) 
        plt.title("IPFB Root Mean Squared residuals (smoothed)",fontsize=20)
        #plt.xlabel("Channel #",fontsize=13)
        plt.ylabel("RMSE",fontsize=16)
        plt.xlabel("Timestream Column Index",fontsize=16)
        plt.tight_layout()
        plt.savefig("img/RMSE_log_virgin_IPFB_residuals_wiener.png")
        plt.show(block=False)
        plt.pause(0.01)
    
    # Chose optimal max iter params
    if delta<=0.3: 
        max_iter10,max_iter5,max_iter3,max_iter1=3,2,2,2
    else: # usually it's 0.5
        max_iter10,max_iter5,max_iter3,max_iter1=18,15,10,5
    # RMS conj gradient descent
    saveas10 = "img/RMSE_conjugate_gradient_descent_10percent.png" if verbose else None
    x_out_10 = conjugate_gradient_descent(B_10, u_10, x0=x0_wiener, rmin=0.0,
            max_iter=max_iter10, k=k, lblock=lblock, verbose=verbose, x_true=x, 
            title="RMSE smoothed gradient steps 10% data salvaged",
            saveas=saveas10)
    # RMS conj gradient descent
    saveas5 = "img/RMSE_conjugate_gradient_descent_5percent.png" if verbose else None
    x_out_5 = conjugate_gradient_descent(B_5, u_5, x0=x0_wiener, rmin=0.0, 
            max_iter=max_iter5, k=k, lblock=lblock, verbose=verbose, x_true=x, 
            title="RMSE smoothed gradient steps 5% data salvaged",
            saveas=saveas5)
    # RMS conj gradient descent
    saveas3 = "img/RMSE_conjugate_gradient_descent_3percent.png" if verbose else None
    x_out_3 = conjugate_gradient_descent(B_3, u_3, x0=x0_wiener, rmin=0.0, 
            max_iter=max_iter3, k=k, lblock=lblock, verbose=verbose, x_true=x, 
            title="RMSE smoothed gradient steps 3% data salvaged",
            saveas=saveas3)
    # RMS conj gradient descent
    saveas1 = "img/RMSE_conjugate_gradient_descent_1percent.png" if verbose else None
    x_out_1 = conjugate_gradient_descent(B_1, u_1, x0=x0_wiener, rmin=0.0, 
            max_iter=max_iter1, k=k, lblock=lblock, verbose=verbose, x_true=x, 
            title="RMSE smoothed gradient steps 1% data salvaged",
            saveas=saveas1)        
    return x0, x0_wiener, x_out_10, x_out_5, x_out_3, x_out_1
 

def reconstruct_long_signal(signal, delta=0.5, verbose=False):
    """PFB's signal, quantizes, then undoes the PFB, returns 
    reconstructed signal using naive inversion, wiener filter, and 
    extra information. 

    We assume, conservatively, that the first 5 and last 5 frames of
    each reconstruction will succumb to edge effects. With this in mind
    we iPFB our signal in batches of 80 * 2048 (k * lblock). 
    
    Parameters
    ----------
    signal : np.ndarray
        Time-series data
    delta : float
        Quantization interval
    """
    k = 80 # number of frames
    lblock = 2048 # size of frames

    # normalize signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    signal_out0 = np.zeros(len(signal))
    signal_out_wiener = np.zeros(len(signal))
    signal_out10 = np.zeros(len(signal))
    signal_out5 = np.zeros(len(signal))
    signal_out3 = np.zeros(len(signal))
    #signal_out1 = np.zeros(len(signal))
    for i in np.arange(0,len(signal) - k*lblock,(k-10)*lblock):
        idxs = np.arange(i,i+k*lblock)
        idxs_no_edge = idxs[5*lblock:-5*lblock] # subset of indices without edge effects present
        x = signal[idxs]
        x0,x_wiener,x10,x5,x3,x1 = conj_grad_one_three_five_perc(
                x,delta,k,lblock,verbose=verbose)
        signal_out0[idxs_no_edge] = x0[5*lblock:-5*lblock]
        signal_out_wiener[idxs_no_edge] = x_wiener[5*lblock:-5*lblock]
        #signal_out1[idxs_no_edge] = x1[5*lblock:-5*lblock]
        signal_out3[idxs_no_edge] = x3[5*lblock:-5*lblock]
        signal_out5[idxs_no_edge] = x5[5*lblock:-5*lblock]
        signal_out10[idxs_no_edge] = x10[5*lblock:-5*lblock]

    print("INFO: Reconstructed signal, serializing npy arrays")
    nowstring=datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\t{nowstring}_signal.npy")
    np.save(f"{nowstring}_signal.npy",np.asarray(signal,dtype="float32"))
    print(f"\t{nowstring}_signal_out0.npy")
    np.save(f"{nowstring}_signal_out0.npy",np.asarray(signal_out0,dtype="float32"))
    print(f"\t{nowstring}_signal_wiener.npy")
    np.save(f"{nowstring}_signal_wiener.npy",np.asarray(signal_out_wiener,dtype="float32"))
    #print(f"\t{nowstring}_signal_out1.npy")
    #np.save(f"{nowstring}_signal_out1.npy",np.asarray(signal_out1,dtype="float32"))
    print(f"\t{nowstring}_signal_out3.npy")
    np.save(f"{nowstring}_signal_out3.npy",np.asarray(signal_out3,dtype="float32"))
    print(f"\t{nowstring}_signal_out5.npy")
    np.save(f"{nowstring}_signal_out5.npy",np.asarray(signal_out5,dtype="float32"))
    print(f"\t{nowstring}_signal_out10.npy")
    np.save(f"{nowstring}_signal_out10.npy",np.asarray(signal_out10,dtype="float32"))
    return signal_out0,signal_out_wiener,signal_out1,signal_out3,signal_out5,signal_out10















