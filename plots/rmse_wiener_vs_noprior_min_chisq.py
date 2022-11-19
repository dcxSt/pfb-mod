print("\nINFO: Running rmse_wiener_vs_noprior_min_chisq.py\n")
"""
To see how much a chi-squared minimization improves the Wiener filter
"""

import sys
sys.path.append("..")
import conjugate_gradient as cg
from matrix_operations import A,A_inv,A_inv_wiener,A_quantize,AT
import pfb
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt


"""Main"""
# Simulate and plot the iPFB quantization noise before and after 
# correction. 

# Choose a photocopy & color-blind friendly colormap
colors=plt.get_cmap('Set2').colors # get list of RGB color values

delta = 0.5     # Quantization interval
k = 80          # Determines length of simulated signal k*lblock
lblock = 2048

# Simulated input data is randomly sampled from a normal distribution. 
x = np.random.normal(0,1,lblock*k) 
# `d` is what the data looks like after it's been through the PFB and it's been quantized. 
d = A_quantize(x,delta) 
# `N_inv` and `Q_inv` are *diagonal* matrices, so we store them as 1D-arrays 
N_inv = np.ones(len(x)) * 6 / delta**2 


"""Construct the chi-square minimization linear operator"""
B = lambda ts: AT(N_inv * A(ts)) 
u = AT(N_inv * d) 

"""Optimize CHI squared using conjugate gradient method."""
# x0 is the standard IPFB reconstruction
x0 = np.real(A_inv(d))
x0_wiener = np.real(A_inv_wiener(d, 0.25)) # Weiner threshold 0.25

# RMS conj gradient descent, optimal x
x_opt = cg.conjugate_gradient_descent(B, u, x0=x0_wiener, rmin=0.0, 
        max_iter=5, k=k, lblock=lblock, verbose=True, x_true=x, 
        title="RMSE smoothed gradient steps, no data salvaged",
        saveas="img/RMSE_conjugate_gradient_descent_0percent.png")        
 

"""Plot and compare result"""
plt.figure(figsize=(14,4))

# rms virgin pfb
rms_virgin = (x - x0)**2
rms_virgin = np.reshape(rms_virgin[5*lblock:-5*lblock],(k-10,lblock)) # bad practice to hard code k=80...? I just want to write this fast
rms_net_virgin = np.sqrt(np.mean(rms_virgin))
rms_virgin = np.sqrt(np.mean(rms_virgin,axis=0))
rms_virgin = cg.mav(rms_virgin,5)
plt.semilogy(rms_virgin[5:-5],label="rmse virgin ipfb",color=colors[0]) 

# rms wiener filtered pfb
rms_wiener = (x - x0_wiener)**2
rms_wiener = np.reshape(rms_wiener[5*lblock:-5*lblock],(k-10,lblock)) 
rms_net_wiener = np.sqrt(np.mean(rms_wiener))
rms_wiener = np.sqrt(np.mean(rms_wiener,axis=0))
plt.semilogy(rms_wiener[5:-5],label="rmse wiener filtered",color=colors[1]) 

# rms chisq optimized
rms_opt = (x - x_opt)**2
rms_opt = np.reshape(rms_opt[5*lblock:-5*lblock],(k-10,lblock))
rms_net_opt = np.sqrt(np.mean(rms_opt))
rms_opt = np.sqrt(np.mean(rms_wiener,axis=0))
plt.semilogy(rms_opt[5:,-5],label="rmse optimized",color=colors[2])

# Plot params
plt.grid(which="both") 
plt.legend()
plt.title("IPFB Root Mean Squared residuals (smoothed)",fontsize=20)
plt.ylabel("RMSE",fontsize=16)
plt.xlabel("Timestream Column Index",fontsize=16)
plt.tight_layout()
plt.savefig("img/RMSE_log_IPFB_residuals_wiener_vs_optimized.png")
plt.show()

   




