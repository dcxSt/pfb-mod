print("\nINFO: Running rmse_conjugate_gradient_plots.py\n")

import sys
sys.path.append("..")
import conjugate_gradient as cg
import pfb
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt

def A(x):
    """Applies PFB, irfft's that, flatten."""
    # Forward PFB the Signal
    b = pfb.forward_pfb(x)
    # Inverse Fourier Transform along axis=1
    b = irfft(b)
    # Apply circulant boundary conditions
    b = np.concatenate([b, b[:3, :]], axis=0)
    return b.flatten()

def A_inv(b_flat):
    """Inverse of A. Reshape the array, rfft, iPFB."""
    # Sanity check
    if len(b_flat)/lblock != len(b_flat)//lblock: 
        raise Exception("Dimensions of input do not match lblock!")
    # Reshape array so that it looks like irfft'd pfb output dims
    b = b_flat.reshape((-1,lblock))[:-3,:]
    # Rfft along axis=1
    b = rfft(b)
    return pfb.inverse_pfb(b)

def A_inv_wiener(b_flat, wiener_thresh=0.25):
    """Inverse of A with wiener filtering. Reshape the array, rfft, iPFB with wiener filter."""
    # Sanity check
    if len(b_flat)/lblock != len(b_flat)//lblock: 
        raise Exception("Dimensions of input do not match lblock!")
    # Reshape array so that it looks like irfft'd pfb output dims
    b = b_flat.reshape((-1,lblock))[:-3,:]
    # Rfft along axis=1
    b = rfft(b)
    return pfb.inverse_pfb(b, wiener_thresh=wiener_thresh)

def A_quantize(x, delta):
    """Takes signal, pfb's it, quantizes, irfft's that."""
    # Forward PFB the signal
    b = pfb.forward_pfb(x)
    # Quantize the filter bank
    # The sqrt is to account for the next IRFFT step
    # b = pfb.quantize(b, np.sqrt(2*(b.shape[1] - 1)) * delta) 
    b = pfb.quantize_8_bit(b, np.sqrt(2*(b.shape[1] - 1)) * delta) 
    # Inverse Fourier Transform
    b = irfft(b) # Same as apply along axis=1
    # Apply circulant boundary conditions
    b = np.concatenate([b, b[:3, :]], axis=0)
    return b.flatten() 

def R(x):
    """Re-ordering matrix (involution)."""
    lx = len(x)
    if lx/lblock != lx//lblock: 
        raise Exception("Len x must divide lblock.")
    k = lx // lblock
    out = np.zeros(lx)
    for i in range(k):
        out[i*lblock:(i+1)*lblock] = x[(k-i-1)*lblock:(k-i)*lblock]
    return out

def AT(x): # the transpose of A
    return R(A(R(x)))


"""Main"""
# Simulate and plot the iPFB quantization noise before and after 
# correction. 

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
_,saved_idxs_5 = cg.get_saved_idxs(5, 0.05, k, lblock)
# The noise matrix for the prior. 
prior_5 = np.zeros(len(x)) # What we know about x, information we salvaged. 
# The data we save will also be 8-bit quantized. 
prior_5[saved_idxs_5] = pfb.quantize_real(x[saved_idxs_5].copy() , delta) # Quantized original signal. 

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

Q_inv_1 = np.zeros(len(x)) 
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
plt.semilogy(rms_virgin[5:-5],label="rmse virgin ipfb") 

# rms wiener filtered pfb
rms_wiener = (x - x0_wiener)**2
rms_wiener = np.reshape(rms_wiener[5*lblock:-5*lblock],(k-10,lblock)) 
rms_net_wiener = np.sqrt(np.mean(rms_wiener))
rms_wiener = np.sqrt(np.mean(rms_wiener,axis=0))
plt.semilogy(rms_wiener[5:-5],label="rmse wiener filtered") 

plt.grid(which="both") 
plt.legend()
plt.title("Log IPFB RMS residuals (smoothed)\nrmse virgin = {:.3f} rmse wiener = {:.3f}".format(rms_net_virgin,rms_net_wiener),fontsize=20) 
plt.xlabel("Channel (n)",fontsize=13)
plt.ylabel("RMSE",fontsize=13)
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
    




