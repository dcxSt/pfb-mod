import sys
sys.path.append("..")
from optimal_wiener_thresh import get_mse_wiener, get_optimal_wiener_thresh
from helper import r_window_to_matrix_eig
import numpy as np
import matplotlib.pyplot as plt

# Get the eigengrid
ntap = 4
lblock = 2048
sinc = np.sinc(np.arange(-ntap/2, ntap/2, 1/lblock))
eigengrid_hann = r_window_to_matrix_eig(sinc * np.hanning(len(sinc)))
print(f"DEBUG: eigengrid mean abs value {np.mean(abs(eigengrid_hann.flatten()))}")

sigma_s = 1.0
sigma_n = 0.5 / np.sqrt(12)
thresh_optimal = get_optimal_wiener_thresh(eigengrid_hann, sigma_s, sigma_n)
rmse_optimal = np.sqrt(get_mse_wiener(eigengrid_hann, sigma_s, sigma_n, thresh_optimal))
rmse_no_filter = np.sqrt(get_mse_wiener(eigengrid_hann, sigma_s, sigma_n, 0.0))
print(f"\nThe optimal threshold is {thresh_optimal}")

plt.subplots(2, 1, figsize=(12,5))
plt.subplot(121)
# Set a cieling, divergence leads to ugly plot
cieling = np.exp(3.0)
rmse_no_filter[np.where(rmse_no_filter > cieling)] = cieling
plt.imshow(np.log(rmse_no_filter.T), aspect="auto", cmap="Set2")
plt.title("No filter", fontsize=16)
plt.colorbar()
plt.xlabel("Channel #",fontsize=14)
plt.ylabel("RFFT Errors on channel",fontsize=14)

plt.subplot(122)
rmse_optimal.T[0,0] = cieling # Scale the colors hack
plt.imshow(np.log(rmse_optimal.T), aspect="auto", cmap="Set2")
plt.title("Wiener filter, optimal threshold value phi={:.2f}".format(thresh_optimal),fontsize=16)
plt.colorbar()
plt.xlabel("Channel #",fontsize=14)
plt.ylabel("RFFT Errors on channel",fontsize=14)

plt.suptitle("Eigenspectrum log RMSE Sinc-Hanning", fontsize=22)
plt.tight_layout()
plt.savefig("img/rmse_wiener_eigenspec.png")
plt.show(block=True)



