print("\nINFO: Running eigenvalues_ntap4_lblock2048.py\n")

# Local import
import sys
sys.path.append("..")
import helper as h
# Libraries
import numpy as np
import matplotlib.pyplot as plt

# `ntap` is the number of taps
# `lblock` is `2*nchan` which is the number of channels before transform
ntap, lblock = 4, 2048

# A sinc array
sinc = np.sinc(np.linspace(-ntap/2, ntap/2, ntap*lblock))
# Generate the matrix of eigenvalues
mat_eig = h.r_window_to_matrix_eig(sinc * np.hanning(ntap*lblock), 
                                    ntap, lblock)

# Plot the figure
plt.figure(figsize=(8.5,7))
#plt.title("Eigenvalues ntap={} lblock={}".format(ntap,lblock),fontsize=22)
plt.imshow(np.abs(mat_eig.T), aspect='auto',cmap="BrBG") # colorblind cmap
plt.title("PFB Eigenvalues (Magnitude)",fontsize=22)
plt.xlabel("Frame Index", fontsize=17)
plt.ylabel("FFT of zero padded Sinc-Hann-derived kernels", fontsize=17)
plt.colorbar()
plt.tight_layout()
# Optionally save the figure
plt.savefig("img/eigenvalues_ntap4_lblock2048.png")
plt.show(block=True)





