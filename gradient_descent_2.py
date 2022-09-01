#!/usr/bin/python3.8
"""
Created on 2021.06.24

Author : Stephen Fay
"""

import helper as h 
from eigenspectrum_plotting_library import image_eig2 
from constants import *
import loss_functions as lf 
import jax.numpy as np
from jax import grad,jit 
import matplotlib.pyplot as plt

# choose learning parameters, these determine what window you end up with
EIG_RATE = 0.1					# float			reccomended : 0.01~0.25
SINC_SIM_RATE = 0.25		    # float			reccomended : 0.01~0.5
SMOOTH_WIDTH = 15				# int 			reccomended : 5~20
SMOOTH_TIMES = 2 				# int 			reccomended : 1~8

# load the loss functions and preload their gradients for efficient computation
loss_eig = lf.loss_eig_hard_thresh_025
grad_eig = jit(grad(loss_eig,argnums=0))
loss_lobes = lf.loss_sidelobes_window_sinc2
grad_lobes = jit(grad(loss_lobes,argnums=0)) 


# initialize window as the standard SINC
x = SINC.copy()
descent_rate_arr = [] # keeps track of how fast we are descending

# do gradient descent
for i in range(100):
	xold = x.copy() # copy the window for keeping track of descent rate

	# descent step 
	x = x - EIG_RATE * grad_eig(x) 
	for j in range(SMOOTH_TIMES): x = h.mav(x,k=SMOOTH_WIDTH)
	x = x - SINC_SIM_RATE * grad_lobes(x) 

	# update descent rate array
	descent_rate = np.sqrt(((xold - x)**2).mean())
	descent_rate_arr.append(descent_rate) 
	
	# print some stats
	if i%15==0:
		print("loss_eig :\t\t{}".format(loss_eig(xold)))
		print("grad_eig norm :\t\t{}".format(np.sqrt((grad_eig(xold)**2).mean())))
		print("loss_lobes :\t\t{}".format(loss_lobes(xold)))
		print("grad_lobes norm :\t{}".format(np.sqrt((grad_lobes(xold)**2).mean())))
		print("descent rate :\t\t{}".format(descent_rate))
		print("----"*17)
		print("\n")

print("\nFinnished gradient descent\n")


print("\nThis plot is how fast the descent was\n")

# display how fast the descent was
plt.figure(figsize=(7,5))
plt.title("descent rate")
plt.semilogy(descent_rate_arr)
plt.xlabel("descent time") 
plt.ylabel("steepness of combined gradients")
plt.grid()
plt.show(block=False)
plt.pause(0.05)

print("\nHere are the eigenvalues:")

image_eig2(x) # display the eigenvalues to user

