"""
Created on 2021.06.10 16:54

Author : Stephen Fay
"""

from jax import numpy as jnp
from jax import grad,jit
# from jax.numpy.fft import rfft,irfft 
import helper as h
from constants import SINC,BOXCAR_R_4X,NTAP,LBLOCK 

#%% Helper functions used in loss functions below

"""softplus is essentially a relu but it's differentiable
	x : jax.numpy ndarray float, is the input
	k : float, determines how curvy the 'relu' is, high means sharp turn
	r : float, this is where the bend in the 'relu' is
"""
softplus = lambda x,k,r:h.ln(1+h.exp((x-r)*k))/k 

softplus_sum = lambda x,k,r:softplus(x,k,r).sum()
softplus_deriv = lambda x,k,r:grad(softplus_sum,argnums=0)(x,k,r)


#%% Tried and tested Differentiable Eigenvalue loss functions
# these loss functions penalize low eigenvalues in different ways

# differentiable eigenvalue loss, suggested by JS
def loss_eig(window,ntap=NTAP,lblock=LBLOCK):
    rft = abs(h.window_to_matrix_eig(window,ntap,lblock))
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
def loss_eig_soft_thresh_025(window, ntap=NTAP, lblock=LBLOCK):
    rft = abs(h.r_window_to_matrix_eig(window,ntap,lblock))
    rft.flatten()
    return (soft_thresh_025(rft)).mean() /0.00025 # normalize
    
# a loss function
def loss_eig_hard_thresh_025(window, ntap=NTAP, lblock=LBLOCK):
    rft = abs(h.r_window_to_matrix_eig(window,ntap,lblock))
    rft.flatten()
    return (hard_thresh_025(rft)).mean() /0.00025 # normalize

# loss function on chebyshev coefficients
def loss_eig_hard_thresh_025_cheb(cheb_tail, sinc=SINC, ntap=NTAP, lblock=LBLOCK):
    cheb_window = h.cheb_win(cheb_tail,len(sinc))
    w = cheb_window * sinc 
    return loss_eig_hard_thresh_025(w, ntap=ntap, lblock=lblock)

# loss function on chebyshev coefficients
# computes number of eigenvalues below 0.25 for a cheb window applied with a hanning window
def loss_eig_hard_thresh_025_cheb_hann(cheb_tail, sinc=SINC, ntap=NTAP, lblock=LBLOCK):
    cheb_window = h.cheb_win(cheb_tail,len(sinc))
    w = cheb_window * sinc * jnp.hanning(len(sinc))
    return loss_eig_hard_thresh_025(w, ntap=ntap, lblock=lblock)

# loss function on chebyshev coefficients, assumes first coeff is 1 and the next n_skip coeffs are 0
# cheb tail specifies rest of coeffs
def loss_eig_hard_thresh_025_cheb_hann_skip(cheb_tail,n_skip,sinc=SINC,ntap=NTAP,lblock=LBLOCK):
    cheb_window = h.cheb_win_skip(cheb_tail,len(sinc),n_skip)
    w = cheb_window * sinc * jnp.hanning(len(sinc))
    return loss_eig_hard_thresh_025(w, ntap=ntap, lblock=lblock)
    

# loss function on chebyshev coefficients
# computes number of eigenvalues below 0.25 for a cheb window applied with a hamming window
def loss_eig_hard_thresh_025_cheb_hamm(cheb_tail, sinc=SINC, ntap=NTAP, lblock=LBLOCK):
    cheb_window = h.cheb_win(cheb_tail,len(sinc))
    w = cheb_window * sinc * jnp.hamming(len(sinc))
    return loss_eig_hard_thresh_025(w, ntap=ntap, lblock=lblock)

# a loss function
def loss_eig_hard_thresh_01(window, ntap=NTAP, lblock=LBLOCK):
    rft = abs(h.r_window_to_matrix_eig(window))
    rft.flatten() # don't think this is necessary
    return (hard_thresh_01(rft)).mean() /0.00025 # normalize


eig_loss_list = [loss_eig,
                 loss_eig_soft_thresh_025,
                 loss_eig_hard_thresh_025,
                 loss_eig_hard_thresh_01]


#%% Tried and tested differentiable sidelobe loss functions
# these loss functions penalize high sidelobes in various ways

def loss_keep_box_up(window, boxhead=h.log10(abs(BOXCAR_R_4X[:13]))):
    """penalizes window if it's boxcar is too low with a softplus"""
    head = abs(h.window_pad_to_box_rfft(window,pad_factor=4.0)[:13])
    log_head = h.log10(head[:13]) # 13 = width of sinc boxcar 
    eps = 0.1
    l = softplus(boxhead - log_head - eps, k=150.0, r=0.0) 
    return l.mean() / 0.01715 

def loss_sidelobes_window_sinc2(window):
    return (((window - SINC)*10)**2).mean() * 27 # 27 is a normalization factor

def loss_sidelobes_window_sinc2_cheb(cheb_tail,sinc=SINC):
    w = h.cheb_win(cheb_tail,len(sinc))
    return loss_sidelobes_window_sinc2(w)
