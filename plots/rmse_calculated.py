print("\nINFO: Running rmse_calculated.py\n")

import sys
sys.path.append("..")
import helper as h
import numpy as np
import matplotlib.pyplot as plt


ntap = 4
lblock = 2048
sinc = np.sinc(np.arange(-ntap/2,ntap/2,1/lblock))

eigengrid = h.r_window_to_matrix_eig(sinc)
eigengrid_hann = h.r_window_to_matrix_eig(sinc * np.hanning(len(sinc)))
eigengrid_hamm = h.r_window_to_matrix_eig(sinc * np.hamming(len(sinc)))

# Plots
plt.figure(figsize=(14,4))
plt.semilogy(np.mean(1/abs(eigengrid**2),axis=1), ".", label="sinc")
plt.semilogy(np.mean(1/abs(eigengrid_hann**2),axis=1), ".", label="sinc hanning")
plt.semilogy(np.mean(1/abs(eigengrid_hamm**2),axis=1), ".", label="sinc hamming")
plt.title("Analytically Derived RMSE on one Chunk of Data", fontsize=20)
plt.ylabel("Log R[n]")
plt.xlabel("n")
plt.grid(which="both")
plt.legend()

plt.savefig("RMSE_analytic_lblock.png")
plt.show(block=True)

