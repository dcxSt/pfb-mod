# Finds the optimal weiner threshold value
# just a bit of plug and play
import numpy as np
from helper import r_window_to_matrix_eig

def get_mse(w_xi, sigma_s, sigma_n, thresh):
    """Returns analytically derived mean-squared-error

    Parameters
    ----------
    w_xi : float or np.ndarray
        A value in the eigenmatrix that is causing us trouble. You can
        also pass the whole eigenmatrix. (Thank you Python!)
    sigma_s : float
        The standard deviation of the signal we are recording, i.e. 
        the root mean squared of the raw electric field signal.
    sigma_n : float
        The standard deviation of the quantization noise.
    thresh : float
        The selected value of the threshold. 

    Returns
    -------
    float or np.ndarray
        The rmse for this particular value of w_xi. 
        If an ndarray is passed for w_xi, it will return and ndarray.
    """
    w_xi_sq = np.array(abs(w_xi)**2, dtype="float64") # Might be complex
    recip = 1 / (thresh**2 + w_xi_sq)**2
    # Error induced on well behaved data points by the filter
    filt_err = sigma_s**2 * thresh**4 * (1 - w_xi_sq)**2
    # Error induced by the quantization amplification
    quant_err = sigma_n**2 * (1 + thresh**2)**2 * w_xi_sq 
    return recip * (filt_err + quant_err)


def get_optimal_wiener_thresh(eigengrid, sigma_s=1.0, sigma_n=0.5/np.sqrt(12)):
    """Finds the optimal threshold value for the weiner filter. 

    Parameters
    ----------
    eigengrid : np.ndarray
        The eigengalues of the rows of the Toepliz matrices. 
    sigma_s : float
        The RMS of our electric field samples. 
    sigma_n : float
        The RMSE induced by the quantization noise. Defaults to 
        assuming that delta=0.5 (the quantization interval), therefore
        the RMSE quantization noise is 0.5/sqrt(12) = 0.1443

    Returns
    -------
    float
        The optimal threshold value.
    """
    # Search for the theshold which minimizes the MSE
    thresholds = np.linspace(0, 1, 100)
    mses = np.zeros(100)
    for idx, thresh in enumerate(thresholds):
        mses[idx] = np.mean(get_mse(
                        eigengrid, sigma_s, sigma_n, thresh).flatten())
    print(f"mses {mses}")
    return thresholds[np.argmin(mses)]


if __name__ == "__main__":
    # Get the eigengrid
    ntap = 4
    lblock = 2048
    sinc = np.sinc(np.arange(-ntap/2, ntap/2, 1/lblock))
    eigengrid_hann = r_window_to_matrix_eig(sinc * np.hanning(len(sinc)))
    print(f"DEBUG: eigengrid mean abs value {np.mean(abs(eigengrid_hann.flatten()))}")

    sigma_s = 1.0
    sigma_n = 0.5 / np.sqrt(12)
    thresh_optimal = get_optimal_wiener_thresh(eigengrid_hann, sigma_s, sigma_n)
    rmse_optimal = np.sqrt(get_mse(eigengrid_hann, sigma_s, sigma_n, thresh_optimal))
    rmse_no_filter = np.sqrt(get_mse(eigengrid_hann, sigma_s, sigma_n, 0.0))
    print(f"\nThe optimal threshold is {thresh_optimal}")

    import matplotlib.pyplot as plt
    plt.subplots(2, 1, figsize=(12,5))
    plt.subplot(121)
    # Set a cieling, divergence leads to ugly plot
    cieling = np.exp(3.0)
    rmse_no_filter[np.where(rmse_no_filter > cieling)] = cieling
    plt.imshow(np.log(rmse_no_filter.T), aspect="auto", cmap="rainbow")
    plt.title("No filter", fontsize=15)
    plt.colorbar()

    plt.subplot(122)
    rmse_optimal.T[0,0] = cieling # Scale the colors hack
    plt.imshow(np.log(rmse_optimal.T), aspect="auto", cmap="rainbow")
    plt.title("Wiener filter, optimal threshold value set to {:.2f}".format(thresh_optimal),
                fontsize=15)
    plt.colorbar()

    plt.suptitle("Eigenspectrum log RMSE Sinc-Hanning", fontsize=24)
    plt.tight_layout()
    plt.show(block=True)













