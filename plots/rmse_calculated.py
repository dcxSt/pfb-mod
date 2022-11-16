print("\nINFO: Running rmse_calculated.py\n")

import sys
sys.path.append("..")
import helper as h
import numpy as np
import matplotlib.pyplot as plt

# Choose a photocopy & color-blind friendly colormap
colors=plt.get_cmap('Set2').colors # get list of RGB color values

ntap = 4
lblock = 2048
sinc = np.sinc(np.arange(-ntap/2,ntap/2,1/lblock))

eigengrid = h.r_window_to_matrix_eig(sinc)
eigengrid_hann = h.r_window_to_matrix_eig(sinc * np.hanning(len(sinc)))
eigengrid_hamm = h.r_window_to_matrix_eig(sinc * np.hamming(len(sinc)))

# Plots
plt.figure(figsize=(14,4))
plt.semilogy(np.mean(1/abs(eigengrid**2),axis=1), ".", label="sinc",color=colors[0])
plt.semilogy(np.mean(1/abs(eigengrid_hann**2),axis=1), ".", label="sinc hanning",color=colors[1])
plt.semilogy(np.mean(1/abs(eigengrid_hamm**2),axis=1), ".", label="sinc hamming",color=colors[2])
#plt.title("Analytically Derived RMSE on one Chunk of Data", fontsize=20)
plt.title("Analytically Derived RMSE amplification per channel", fontsize=20)
#plt.ylabel("Log R[n]")
plt.ylabel("Gain on quantization noise",fontsize=16)
plt.xlabel("Timestream Column Index",fontsize=16)
plt.grid(which="both")
plt.legend()
plt.tight_layout()
plt.savefig("img/RMSE_analytic_lblock.png")
plt.show(block=True)


