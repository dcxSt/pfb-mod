import pfb
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt

#%% Helper methods / routines

def get_saved_idxs(n_per_save=5, prct=0.03, k=80, lblock=2048):
    """Returns 3% of indices of the most troublesome channels.

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


# TODO
def mcmc(B, u, x0=None, rmin=0.1, max_iter=20, 
        k=80, lblock=2048, verbose=False, x_true=None): 
    """Minimize chi-squared by casting as gradient descent problem. 
    
    This conjugate gradient descent method approximately solves the 
    linear equasion Bx = u. B is a symmetric, positive definite matrix
    operator and u is the target. 
    
    Parameters
    ----------
    B : callable
        A function which is actually a square matrix in disguise. 
    u : np.ndarray
        Data vector. The rhs of the quadratic equasion we are solving. 
    x0 : np.ndarray
        Initial guess for x. 
    rmin : float
        The threshold value for stopping the descent. If the RMSE goes
        below this value, we're gucci. 
    max_iter : int
        The maximal number of descent iterations to make before stopping. 
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
    print("INFO: Conjugate Gradient descent completed.") 
    return x




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
        A function which is actually a square matrix in disguise. 
    u : np.ndarray
        Data vector. The rhs of the quadratic equasion we are solving. 
    x0 : np.ndarray
        Initial guess for x. 
    rmin : float
        The threshold value for stopping the descent. If the RMSE goes
        below this value, we're gucci. 
    max_iter : int
        The maximal number of descent iterations to make before stopping. 
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
    print("INFO: Conjugate Gradient descent completed.") 
    return x














