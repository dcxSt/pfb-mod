"""
Created on 2021.06.03 16:08:57

Author : Stephen Fay
"""

import numpy as np
from scipy.fft import fft,ifft,fftshift,ifftshift
from constants import *

#%% fourier transforms
def window_to_box(window):
    return fftshift(fft(fftshift(window))) # gets taller

def box_to_window(box):
    return ifftshift(ifft(ifftshift(box))) # gets smaller

#%% basic windows, for all windows see windows.py

def sinc_window(ntap=NTAP,lblock=LBLOCK):
    # i don't like np.arange, it's vulnerable to failur, 
    # should be linspace, but this is how R.S. implemented it, and it'll give a (tiny bit) different array
    return np.sinc(np.arange(-ntap/2,ntap/2,1/lblock)) 

def sinc_hamming(ntap=NTAP,lblock=LBLOCK):
    return np.hamming(ntap*lblock) * sinc_window(ntap,lblock)

#%% helpers for eigenvalue displays
def chop_win(w,ntap=4,lblock=2048):
    """Chop lblock bits of len ntap of window to get ready for DFT"""
    if ntap*lblock!=len(w):raise Exception("len window incompatible")
    return np.reshape(w,(ntap,lblock)).T

def zero_padding(w2d,n_zeros=1024):
    pad = np.zeros((len(w2d),n_zeros))
    return np.concatenate([w2d,pad],axis=1)


def window_to_matrix_eig(w,ntap=NTAP,lblock=LBLOCK):
    w2d = chop_win(w,ntap,lblock)
    w2d_padded = zero_padding(w2d)
    ft = np.apply_along_axis(fft,1,w2d_padded)
    return ft
def matrix_eig_to_window(ft_w2d,ntap=NTAP):
    w2d_padded = np.apply_along_axis(ifft,1,ft_w2d)
    w2d = w2d_padded[:,:ntap]
    return np.real(np.concatenate(w2d.T))
def matrix_eig_to_window_complex(ft_w2d,ntap=NTAP):
    w2d_padded = np.apply_along_axis(ifft,1,ft_w2d)
    w2d = w2d_padded[:,:ntap]
    return np.concatenate(w2d.T)