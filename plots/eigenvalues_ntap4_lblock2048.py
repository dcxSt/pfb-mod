print("\nINFO: Running eigenvalues_ntap4_lblock2048.py\n")

# Local import
import sys
sys.path.append("..")
import helper as h
# Libraries
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

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
plt.imshow(np.abs(mat_eig.T), aspect='auto', origin='lower', cmap="viridis",
           extent=[0, mat_eig.shape[0], 0, 0.5]) # colorblind cmap
plt.xlabel("Frame Index", fontsize=17)
plt.ylabel("Frequency (cycles/sample)", fontsize=17)
plt.colorbar()
plt.tight_layout()
# Optionally save the figure
plt.savefig("img/eigenvalues_ntap4_lblock2048.png",dpi=650)
plt.show(block=True)





