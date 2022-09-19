print("\nINFO: Running rmse_wiener.py\n")

import sys
sys.path.append("..")
import pfb
from conjugate_gradient import mav
import numpy as np
import matplotlib.pyplot as plt

# Global variables
lblock = 2048
nchan = 1025
ntap = 4


def simulate_quantization_error_wiener( delta=0.5 , n_sims = 100 , k=60 , wiener_thresh=0.1 ):
    # Rem: the RMSE is delta / sqrt(12)
    # input param delta is the quantization step
    x_arr = [] # list of actual input timestream arrays
    x_ipfb_arr = [] # list of IPFB output arrays
    x_wiener_arr = [] # list of IPFB and wiener arrays
    print("INFO: Randomly generating data quantization-error simulations.")
    for sim in range(n_sims):
        # if sim==0:print("Simulation Number __ out of {}:".format(n_sims))
        # if sim%1==0: print(sim,end=" ")
        x = np.random.normal(0,1,lblock*k) 
        
        d = pfb.forward_pfb(x) 
        # Quantize the pfb 
        # The square root is to normalize, because the rfft doesn't do 
        # it by default. 
        d = pfb.quantize_8_bit( d , np.sqrt(2*(d.shape[1] - 1)) * delta ) 
                
        x_ipfb = pfb.inverse_pfb(d) # inver the pfb
        x_wiener = pfb.inverse_pfb(d,wiener_thresh=wiener_thresh)
        
        # save the arrays
        x_arr.append(x) 
        x_ipfb_arr.append(x_ipfb) 
        x_wiener_arr.append(x_wiener)
        
    return np.array(x_arr), np.array(x_ipfb_arr), np.array(x_wiener_arr)


def get_rmse(se):
    se = se[:,5*lblock:-5*lblock] # Drop error prone edge-effects
    se = se.reshape((n_sims*(k-10), lblock))
    rmse = np.sqrt(np.mean(se, axis=0)) # Calculate the rmse
    return rmse


#%% Main

# Parameters
k = 80
n_sims = 20
wiener_thresh = 0.5

# Simulate random noise
x, x_virgin, x_wiener = simulate_quantization_error_wiener(n_sims=n_sims,
                            k=k, wiener_thresh=wiener_thresh)

# Process simulated quantization error
se_virgin = np.real(x - x_virgin)**2 # squared error virgin
se_wiener = np.real(x - x_wiener)**2 # squared error after wiener filtering
se_virgin = se_virgin[:, 5*lblock:-5*lblock] # Chop off dirty part
se_wiener = se_wiener[:, 5*lblock:-5*lblock] # Chop off dirty part
rmse_virgin = np.sqrt(np.mean(se_virgin, axis=0))
rmse_wiener = np.sqrt(np.mean(se_wiener, axis=0))
# Average over a single LBLOCK chunk
se_virgin_avg = se_virgin.reshape((n_sims * (k-10), lblock))
se_wiener_avg = se_wiener.reshape((n_sims * (k-10), lblock))
rmse_virgin_avg = np.sqrt(np.mean(se_virgin_avg, axis=0))
rmse_wiener_avg = np.sqrt(np.mean(se_wiener_avg, axis=0))

# Plots
# Plot avg RMSE over one lblock sized chunk
plt.subplots(figsize=(14,7))

plt.subplot(211)
plt.plot(rmse_virgin_avg[:20*lblock], label="no filter")
plt.plot(rmse_wiener_avg[:20*lblock], label="wiener filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE")
plt.grid(which="both")

plt.subplot(212)
plt.semilogy(rmse_virgin_avg[:20*lblock], label="no filter")
plt.semilogy(rmse_wiener_avg[:20*lblock], label="wiener filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE Log scale")
plt.grid(which="both")

plt.tight_layout()
plt.savefig("img/RMSE_wiener_lblock.png")

plt.show(block=True)



# Plot estimated RMSE over one lblock sized chunk
plt.subplots(figsize=(14,7))

plt.subplot(211)
plt.plot(rmse_virgin[:20*lblock], label="no filter")
plt.plot(rmse_wiener[:20*lblock], label="wiener filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE")
plt.grid(which="both")

plt.subplot(212)
plt.semilogy(rmse_virgin[:20*lblock], label="no filter")
plt.semilogy(rmse_wiener[:20*lblock], label="wiener filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE Log scale")
plt.grid(which="both")

plt.tight_layout()
plt.savefig("img/RMSE_wiener_long_time.png")

plt.show(block=True)


