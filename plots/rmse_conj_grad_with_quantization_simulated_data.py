print("\nINFO: Running rmse_conj_grad_with_quantization_simulated_data.py\n")

import sys
sys.path.append("..")
import conjugate_gradient as cg
import numpy as np

   
"""Main"""
# Simulate and plot the iPFB quantization noise before and after 
# correction. 

delta = 0.5     # Quantization interval
k = 80          # Determines length of simulated signal k*lblock
lblock = 2048

# Simulated input data is randomly sampled from a normal distribution. 
x = np.random.normal(0,1,lblock*k) 
_ = cg.conj_grad_one_three_five_perc(x, delta, k, lblock, True)




