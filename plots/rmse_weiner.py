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


def simulate_quantization_error_weiner( delta=0.5 , n_sims = 100 , k=60 , weiner_thresh=0.1 ):
    # input param delta is the quantization step
    x_arr = [] # list of actual input timestream arrays
    x_ipfb_arr = [] # list of IPFB output arrays
    x_weiner_arr = [] # list of IPFB and weiner arrays
    print("INFO: Randomly generating data quantization-error simulations.")
    for sim in range(n_sims):
        # if sim==0:print("Simulation Number __ out of {}:".format(n_sims))
        # if sim%1==0: print(sim,end=" ")
        x = np.random.normal(0,1,lblock*k) 
        
        d = pfb.forward_pfb(x) 
        d = pfb.quantize_8_bit( d , np.sqrt(2*(d.shape[1] - 1)) * delta ) # quantize the pfb 
                
        x_ipfb = pfb.inverse_pfb(d) # inver the pfb
        x_weiner = pfb.inverse_pfb(d,weiner_thresh=weiner_thresh)
        
        # save the arrays
        x_arr.append(x) 
        x_ipfb_arr.append(x_ipfb) 
        x_weiner_arr.append(x_weiner)
        
    return np.array(x_arr), np.array(x_ipfb_arr), np.array(x_weiner_arr)


def get_rmse(se):
    se = se[:,5*lblock:-5*lblock] # Drop error prone edge-effects
    se = se.reshape((n_sims*(k-10), lblock))
    rmse = np.sqrt(np.mean(se, axis=0)) # Calculate the rmse
    return rmse


#%% Main

# Parameters
k = 80
n_sims = 20
weiner_thresh = 0.5

# Simulate random noise
x, x_virgin, x_weiner = simulate_quantization_error_weiner(n_sims=n_sims,
                            k=k, weiner_thresh=weiner_thresh)

# Process simulated quantization error
se_virgin = np.real(x - x_virgin)**2 # squared error virgin
se_weiner = np.real(x - x_weiner)**2 # squared error after weiner filtering
se_virgin = se_virgin[:, 5*lblock:-5*lblock] # Chop off dirty part
se_weiner = se_weiner[:, 5*lblock:-5*lblock] # Chop off dirty part
rmse_virgin = np.sqrt(np.mean(se_virgin, axis=0))
rmse_weiner = np.sqrt(np.mean(se_weiner, axis=0))
# Average over a single LBLOCK chunk
se_virgin_avg = se_virgin.reshape((n_sims * (k-10), lblock))
se_weiner_avg = se_weiner.reshape((n_sims * (k-10), lblock))
rmse_virgin_avg = np.sqrt(np.mean(se_virgin_avg, axis=0))
rmse_weiner_avg = np.sqrt(np.mean(se_weiner_avg, axis=0))

# Plots
# Plot avg RMSE over one lblock sized chunk
plt.subplots(figsize=(14,7))

plt.subplot(211)
plt.plot(rmse_virgin_avg[:20*lblock], label="no filter")
plt.plot(rmse_weiner_avg[:20*lblock], label="weiner filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE")
plt.grid(which="both")

plt.subplot(212)
plt.semilogy(rmse_virgin_avg[:20*lblock], label="no filter")
plt.semilogy(rmse_weiner_avg[:20*lblock], label="weiner filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE Log scale")
plt.grid(which="both")

plt.tight_layout()
plt.savefig("RMSE_weiner_lblock.png")

plt.show(block=True)



# Plot estimated RMSE over one lblock sized chunk
plt.subplots(figsize=(14,7))

plt.subplot(211)
plt.plot(rmse_virgin[:20*lblock], label="no filter")
plt.plot(rmse_weiner[:20*lblock], label="weiner filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE")
plt.grid(which="both")

plt.subplot(212)
plt.semilogy(rmse_virgin[:20*lblock], label="no filter")
plt.semilogy(rmse_weiner[:20*lblock], label="weiner filter")
plt.legend()
plt.xlabel("Time, in number of samples")
plt.title("RMSE Log scale")
plt.grid(which="both")

plt.tight_layout()
plt.savefig("RMSE_weiner_long_time.png")

plt.show(block=True)


