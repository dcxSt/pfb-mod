print("\nINFO: Running rmse_conjugate_gradient_plots.py\n")

import sys
sys.path.append("..")
import conjugate_gradient as cg
from matrix_operators import A,A_inv,A_inv_wiener,A_quantize,AT
from pfb import quantize_real
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt


"""Main"""
# Simulate and plot the iPFB quantization noise before and after 
# correction. 

# Choose a photocopy & color-blind friendly colormap
colors=plt.get_cmap('Set2').colors # get list of RGB color values

delta = 0.5     # Quantization interval
delta_in = 0.5  # Quantization interval of the input
k = 80          # Determines length of simulated signal k*lblock
lblock = 2048

# Simulated input data is randomly sampled from a normal distribution. 
x = np.random.normal(0,1,lblock*k) 
# Quantize x with ADC
x = quantize_real(x,delta_in)
# `d` is what the data looks like after it's been through the PFB and it's been quantized. 
d = A_quantize(x,delta) 

# ---depricated---
## `N_inv` and `Q_inv` are *diagonal* matrices, so we store them as 1D-arrays 
#N_inv = np.ones(len(x)) * 6 / delta**2 
# ---depricated---

"""5 percent of original data given as prior."""
# Get the indices for all the data points we 'salvage' in data collection
_,saved_idxs_5 = cg.get_saved_idxs(5, 0.05, k, lblock)
B_5,u_5 = cg.get_Bu(x,d,saved_idxs_5,delta)

"""3 percent of original data given as prior."""
_,saved_idxs_3 = cg.get_saved_idxs(6, 0.03, k, lblock)
B_3,u_3=cg.get_Bu(x,d,saved_idxs_3,delta)

"""1 percent of original data given as prior."""
_,saved_idxs_1 = cg.get_saved_idxs(7, 0.01, k, lblock)
B_1,u_1=cg.get_Bu(x,d,saved_idxs_1,delta)

"""Optimize CHI squared using conjugate gradient method."""
# x0 is the standard IPFB reconstruction
x0 = np.real(A_inv(d))
x0_wiener = np.real(A_inv_wiener(d, 0.25)) # Weiner threshold 0.25

#"""Plot RMSE of wiener"""
## print("\n\nd={}".format(d)) # trace, they are indeed real
## print("\n\nx_0={}".format(x0)) # complex dtype but zero imag componant
#
## print("\nConjugate Gradient Descent, with 3% extra data (prior is a quantized 3% of original timestream)")
#plt.figure(figsize=(14,4))
#
## rms virgin pfb
#rms_virgin = (x - x0)**2
#rms_virgin = np.reshape(rms_virgin[5*lblock:-5*lblock],(k-10,lblock)) # bad practice to hard code k=80...? I just want to write this fast
#rms_net_virgin = np.sqrt(np.mean(rms_virgin))
#rms_virgin = np.sqrt(np.mean(rms_virgin,axis=0))
#rms_virgin = cg.mav(rms_virgin,5)
#plt.semilogy(rms_virgin[5:-5],label="rmse virgin ipfb",color=colors[0]) 
#
## rms wiener filtered pfb
#rms_wiener = (x - x0_wiener)**2
#rms_wiener = np.reshape(rms_wiener[5*lblock:-5*lblock],(k-10,lblock)) 
#rms_net_wiener = np.sqrt(np.mean(rms_wiener))
#rms_wiener = np.sqrt(np.mean(rms_wiener,axis=0))
#plt.semilogy(rms_wiener[5:-5],label="rmse wiener filtered",color=colors[1]) 
#
#plt.grid(which="both") 
#plt.legend()
##plt.title("Log IPFB RMS residuals (smoothed)\nrmse virgin = {:.3f} rmse wiener = {:.3f}".format(rms_net_virgin,rms_net_wiener),fontsize=20) 
#plt.title("IPFB Root Mean Squared residuals (smoothed)",fontsize=20)
##plt.xlabel("Channel #",fontsize=13)
#plt.ylabel("RMSE",fontsize=16)
#plt.xlabel("Timestream Column Index",fontsize=16)
#plt.tight_layout()
#plt.savefig("img/RMSE_log_virgin_IPFB_residuals_wiener.png")
#plt.show()

def plot_rms(x_out,label):
    rms = (x - x_out)**2
    rms = np.reshape(rms[5*lblock:-5*lblock], (k-10, lblock))
    rms_net = np.sqrt(np.mean(rms)) # net (or total) rms
    rms = np.sqrt(np.mean(rms, axis=0))
    rms_smoothed = cg.mav(rms, 20)[20:-20] # Chop off spoiled values
    plt.plot(rms_smoothed, label=f"{label}")
    return 


# RMS conj gradient descent
x_out_5 = cg.conjugate_gradient_descent(B_5, u_5, x0=x0_wiener, rmin=0.0, 
        max_iter=15, k=k, lblock=lblock, verbose=True, x_true=x, 
        title="RMSE smoothed gradient steps 5% data salvaged",
        saveas="img/RMSE_conjugate_gradient_descent_5percent.png")

# quantize once, look at RMSE
x_out_5_q1 = quantize_real(x_out_5,delta_in)
x_out_5_v1 = cg.conjugate_gradient_descent(B_5, u_5, x0=x_out_5_q1, rmin=0.0,
        max_iter=2, k=k, lblock=lblock, verbose=True, x_true=x,
        title="RMSE smoothed gradient steps 5% data salvaged")
x_out_5_q2 = quantize_real(x_out_5,delta_in)
x_out_5_v2 = cg.conjugate_gradient_descent(B_5, u_5, x0=x_out_5_q2, rmin=0.0,
        max_iter=2, k=k, lblock=lblock, verbose=True, x_true=x,
        title="RMSE smoothed gradient steps 5% data salvaged")
x_out_5_q3 = quantize_real(x_out_5,delta_in)
x_out_5_v3 = cg.conjugate_gradient_descent(B_5, u_5, x0=x_out_5_q3, rmin=0.0,
        max_iter=2, k=k, lblock=lblock, verbose=True, x_true=x,
        title="RMSE smoothed gradient steps 5% data salvaged")

plot_rms(x_out_5,"descent 0")
plot_rms(x_out_5_q1,"quantized 1")
plot_rms(x_out_5_v1,"descent 1")
plot_rms(x_out_5_q2,"quantized 2")
plot_rms(x_out_5_v2,"descent 2")
plot_rms(x_out_5_q3,"quantized 3")
plot_rms(x_out_5_v3,"descent 3")
plt.legend()
plt.title("RMS descent re-quantize")
plt.tight_layout()
plt.savefig("img/double_descent_rmse.png")
plt.show()




## RMS conj gradient descent
#x_out_3 = cg.conjugate_gradient_descent(B_3, u_3, x0=x0_wiener, rmin=0.0, 
#        max_iter=10, k=k, lblock=lblock, verbose=True, x_true=x, 
#        title="RMSE smoothed gradient steps 3% data salvaged",
#        saveas="img/RMSE_conjugate_gradient_descent_3percent.png")
## RMS conj gradient descent
#x_out_1 = cg.conjugate_gradient_descent(B_1, u_1, x0=x0_wiener, rmin=0.0, 
#        max_iter=5, k=k, lblock=lblock, verbose=True, x_true=x, 
#        title="RMSE smoothed gradient steps 1% data salvaged",
#        saveas="img/RMSE_conjugate_gradient_descent_1percent.png")        
    




