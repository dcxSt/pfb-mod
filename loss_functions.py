"""
Created on 2021.06.10 16:54

Author : Stephen Fay
"""

import jax.numpy as np
from jax import grad,jit
from jax.numpy.fft import rfft,irfft 
import helper as h
from constants import *

#%% Helper functions used in loss functions below

"""softplus is essentially a relu but it's differentiable
	x : jax.numpy ndarray float, is the input
	k : float, determines how curvy the 'relu' is, high means sharp turn
	r : float, this is where the bend in the 'relu' is
"""
softplus = lambda x,k,r:h.ln(1+h.exp((x-r)*k))/k 


#%% Tried and tested Differentiable Eigenvalue loss functions
# these loss functions penalize low eigenvalues in different ways
# differentiable eigenvalue loss
def loss_eig(window,ntap=4,lblock=2048):
    rft = abs(h.window_to_matrix_eig(window))
    quality = rft / (0.1+rft) # the higher this number, the better
    return -quality.mean() / 2.47

# continuous relu function, penalizes values under thresh
def soft_thresh_025(arr1d):
    # every element under thresh is penalized linearly
    return softplus(-arr1d,k=10.0,r=-0.1)

def hard_thresh_025(arr1d):
    # every element under thresh is penalized linearly
    return softplus(-arr1d,k=100.0,r=-0.25)

def hard_thresh_01(arr1d):
    # every element under thresh is penalized linearly
    return softplus(-arr1d,k=50.0,r=-0.1)

# a loss function
def loss_eig_soft_thresh_025(window, ntap=4, lblock=2048):
    rft = abs(h.r_window_to_matrix_eig(window))
    rft.flatten()
    return (soft_thresh_025(rft)).mean() /0.25 # normalize
    
# a loss function
def loss_eig_hard_thresh_025(window, ntap=4, lblock=2048):
    rft = abs(h.r_window_to_matrix_eig(window))
    rft.flatten()
    return (hard_thresh_025(rft)).mean() /0.25 # normalize

# a loss function
def loss_eig_hard_thresh_01(window, ntap=4, lblock=2048):
    rft = abs(h.r_window_to_matrix_eig(window))
    rft.flatten() # don't think this is necessary
    return (hard_thresh_01(rft)).mean() /0.25 # normalize


eig_loss_list = [loss_eig,
                 loss_eig_soft_thresh_025,
                 loss_eig_hard_thresh_025,
                 loss_eig_hard_thresh_01]


#%% Tried and tested 