print("\nINFO: Running rmse_conj_grad_with_quantization_radio_data.py\n")

import sys
sys.path.append("..")
import conjugate_gradient as cg
import numpy as np
import matplotlib.pyplot as plt

   
"""Main"""
# Simulate and plot the iPFB quantization noise before and after 
# correction. 

delta = 0.5     # Quantization interval
k = 80          # Determines length of simulated signal k*lblock
lblock = 2048

# Simulated input data is randomly sampled from a normal distribution. 
x = np.random.normal(0,1,lblock*k) 

# load radio data
radio_data_path="/Users/steve/Documents/code/pfb-mod/data/gqrx_20230412_225548_101000000_2500000_fc.raw" # path to radio data
signal = np.fromfile(radio_data_path, dtype="float32")
# normalize it
signal = (signal - np.mean(signal)) / np.std(signal)

# trunkate signal (temporary)
signal = signal[:500000*k*lblock]

ms_wiener_chunks=[]
ms5_chunks=[]
ms3_chunks=[]
ms1_chunks=[]
print("INFO: Looping through radio data many times over")
# loop through chunks of size k * lblock with 50% overlap
for i in np.arange(0, len(signal) - k*lblock, k*lblock):
    x = signal[i:i+k*lblock]
    x0,x_wiener,xout5,xout3,xout1 = cg.conj_grad_one_three_five_perc(
            x, delta, k, lblock, False)
    # rms wiener filtered pfb
    ms_wiener = (x - x_wiener)**2
    ms_wiener = np.reshape(ms_wiener[5*lblock:-5*lblock],(k-10,lblock))
    ms_wiener_chunks.append(np.mean(ms_wiener,axis=0))
    # rms xout5
    ms5 = (x - xout5)**2
    ms5 = np.reshape(ms5[5*lblock:-5*lblock],(k-10,lblock))
    ms5_chunks.append(np.mean(ms5,axis=0))
    # rms xout3
    ms3 = (x - xout3)**2
    ms3 = np.reshape(ms3[5*lblock:-5*lblock],(k-10,lblock))
    ms3_chunks.append(np.mean(ms3,axis=0))
    # rms xout5
    ms1 = (x - xout1)**2
    ms1 = np.reshape(ms1[5*lblock:-5*lblock],(k-10,lblock))
    ms1_chunks.append(np.mean(ms1,axis=0))



# compute means squared errors
ms_wiener = np.mean(np.asarray(ms_wiener_chunks),axis=0)
ms5 = np.mean(np.asarray(ms5_chunks),axis=0)
ms3 = np.mean(np.asarray(ms3_chunks),axis=0)
ms1 = np.mean(np.asarray(ms1_chunks),axis=0)


plt.figure(figsize=(8,5))
idxs = np.arange(k*lblock//2,k*lblock//2+3*lblock)
#plt.plot((x0-x)[idxs],label="iPFB no filtering")
#plt.plot(np.convolve(rms_wiener,np.hanning(10),"valid"),label="Wiener")
plt.plot(np.sqrt(ms_wiener),label="Wiener")
#plt.plot(,label="1%")
#plt.plot(,label="3%")
#plt.plot(np.convolve(rms5,np.hanning(10),"valid"),label="5%")
plt.plot(np.sqrt(ms5),linewidth=1,label="5%")
plt.plot(np.sqrt(ms3),label="3%")
plt.plot(np.sqrt(ms1),label="1%")
plt.legend()
plt.title("Rutherford Roof Radio, Esimated RMS Error")
plt.xlabel("Index in one Frame")
plt.ylabel("Estimated RMSE")
plt.tight_layout()
plt.savefig("img/rms_radio_data.png",dpi=300)
plt.show()

print("INFO: Done")



