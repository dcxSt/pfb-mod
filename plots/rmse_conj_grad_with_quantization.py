print("\nINFO: Running rmse_conj_grad_with_quantization.py\n")

import sys
sys.path.append("..")
import conjugate_gradient as cg
from matrix_operators import A,A_inv,A_inv_wiener,A_quantize,AT
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

"""5 percent of original data given as prior."""
# Get the indices for all the data points we 'salvage' in data collection
_,saved_idxs_5 = cg.get_saved_idxs(5, 0.05, k, lblock)
# The noise matrix for the prior. 
prior_5 = np.ones(len(x)) # What we know about x, information we salvaged. 
# The data we save will also be 8-bit quantized. 
prior_5[saved_idxs_5] = pfb.quantize_real(x[saved_idxs_5].copy() , delta) # Quantized original signal. 

# Should this be zero?
Q_inv_5 = np.ones(len(x)) # this is a prior, change to zeros if you want zero for infinite uncertainty
Q_inv_5[saved_idxs_5] = np.ones(len(saved_idxs_5)) * (12 / delta**2) # 8 bits per real number (finer std because no complex) 

B_5 = lambda ts: AT(N_inv * A(ts)) + Q_inv_5 * ts # think ts===x
u_5 = AT(N_inv * d) + Q_inv_5 * prior_5 # this is same as mult prior by var=12/delta^2


"""3 percent of original data given as prior."""
_,saved_idxs_3 = cg.get_saved_idxs(6, 0.03, k, lblock)
# The noise matrix for the prior. 
prior_3 = np.zeros(len(x)) # What we know about x, information we salvaged. 
# The data we save will also be 8-bit quantized. 
prior_3[saved_idxs_3] = pfb.quantize_real(x[saved_idxs_3].copy() , delta) # Quantized original signal. 

Q_inv_3 = np.ones(len(x)) # this is a prior, change to zeros if you want zero for infinite uncertainty
Q_inv_3[saved_idxs_3] = np.ones(len(saved_idxs_3)) * (12 / delta**2) # 8 bits per real number (finer std because no complex) 

B_3 = lambda ts: AT(N_inv * A(ts)) + Q_inv_3 * ts # think ts===x
u_3 = AT(N_inv * d) + Q_inv_3 * prior_3 # this is same as mult prior by var=12/delta^2

"""1 percent of original data given as prior."""
_,saved_idxs_1 = cg.get_saved_idxs(7, 0.01, k, lblock)
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
x0_wiener = np.real(A_inv_wiener(d, 0.25)) # Weiner threshold 0.25

# print("\n\nd={}".format(d)) # trace, they are indeed real
# print("\n\nx_0={}".format(x0)) # complex dtype but zero imag componant

# print("\nConjugate Gradient Descent, with 3% extra data (prior is a quantized 3% of original timestream)")
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

plt.grid(which="both") 
plt.legend()
#plt.title("Log IPFB RMS residuals (smoothed)\nrmse virgin = {:.3f} rmse wiener = {:.3f}".format(rms_net_virgin,rms_net_wiener),fontsize=20) 
plt.title("IPFB Root Mean Squared residuals (smoothed)",fontsize=20)
#plt.xlabel("Channel #",fontsize=13)
plt.ylabel("RMSE",fontsize=16)
plt.xlabel("Timestream Column Index",fontsize=16)
plt.tight_layout()
plt.savefig("img/RMSE_log_virgin_IPFB_residuals_wiener.png")
plt.show()

# RMS conj gradient descent
x_out_5 = cg.conjugate_gradient_descent(B_5, u_5, x0=x0_wiener, rmin=0.0, 
        max_iter=15, k=k, lblock=lblock, verbose=True, x_true=x, 
        title="RMSE smoothed gradient steps 5% data salvaged",
        saveas="img/RMSE_conjugate_gradient_descent_5percent.png")
# RMS conj gradient descent
x_out_3 = cg.conjugate_gradient_descent(B_3, u_3, x0=x0_wiener, rmin=0.0, 
        max_iter=10, k=k, lblock=lblock, verbose=True, x_true=x, 
        title="RMSE smoothed gradient steps 3% data salvaged",
        saveas="img/RMSE_conjugate_gradient_descent_3percent.png")
# RMS conj gradient descent
x_out_1 = cg.conjugate_gradient_descent(B_1, u_1, x0=x0_wiener, rmin=0.0, 
        max_iter=5, k=k, lblock=lblock, verbose=True, x_true=x, 
        title="RMSE smoothed gradient steps 1% data salvaged",
        saveas="img/RMSE_conjugate_gradient_descent_1percent.png")        
    




