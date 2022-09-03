"""
Created on 2021.06.03 16:08:57
Author : Stephen Fay

Some helper functions
"""

from constants import SINC,BOXCAR_4X_HEIGHT
import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift, rfft, irfft


#%% fourier transforms and their inverses
def window_to_box(window):
    return fftshift(fft(fftshift(window))) # gets taller

def box_to_window(box):
    return ifftshift(ifft(ifftshift(box))) # gets smaller

def window_pad_to_box(window,pad_factor=4.0):
    # pad the window then fft
    padded_window = np.concatenate([window,np.zeros(int(len(window)*pad_factor))])
    return window_to_box(padded_window) 

def box_to_window_pad(large_box,len_win):
    return box_to_window(large_box)[:len_win]

def window_pad_to_box_rfft(window,pad_factor=4.0):
    padded_window = np.concatenate([window,np.zeros(int(len(window)*pad_factor))])
    return rfft(fftshift(padded_window))

def box_to_window_pad_rfft(large_box,len_win):
    return ifftshift(irfft(large_box))[:len_win]

#%% basic windows

def sinc_window(ntap=4, lblock=2048):
    # i don't like np.arange, it's vulnerable to failure, 
    # should be linspace, but this is how R.S. implemented it, and it'll give a (tiny bit) different array
    return np.sinc(np.arange(-ntap/2,ntap/2,1/lblock)) 

def sinc_hanning(ntap=4, lblock=2048):
    # different form sinc_window and sinc_hamming, uses numpy not jax.numpy
    return np.hanning(ntap*lblock) * np.sinc(np.linspace(-ntap/2, ntap/2, lblock*ntap))

def sinc_hamming(ntap=4, lblock=2048):
    return np.hamming(ntap*lblock) * sinc_window(ntap, lblock)

#%% helpers for eigenvalue displays
def chop_win(w, ntap=4, lblock=2048):
    """Chop lblock bits of len ntap of window to get ready for DFT"""
    if ntap*lblock!=len(w):raise Exception("len window incompatible")
    return np.reshape(w, (ntap, lblock)).T

def zero_padding(w2d, n_zeros=1024):
    pad = np.zeros((len(w2d), n_zeros))
    return np.concatenate([w2d, pad], axis=1)


def window_to_matrix_eig(w, ntap=4, lblock=2048, zero_pad_len=1024):
    w2d = chop_win(w, ntap, lblock)
    w2d_padded = zero_padding(w2d, zero_pad_len)
    ft = np.apply_along_axis(fft, 1, w2d_padded)
    return ft

def matrix_eig_to_window(ft_w2d, ntap=4):
    w2d_padded = np.apply_along_axis(ifft, 1, ft_w2d)
    w2d = w2d_padded[:, :ntap]
    return np.real(np.concatenate(w2d.T))

def matrix_eig_to_window_complex(ft_w2d, ntap=4):
    w2d_padded = np.apply_along_axis(ifft, 1, ft_w2d)
    w2d = w2d_padded[:, :ntap]
    return np.concatenate(w2d.T)

def r_window_to_matrix_eig(w, ntap=4, lblock=2048, zero_pad_len=1024):
    w2d = chop_win(w, ntap, lblock)
    w2d_padded = zero_padding(w2d, zero_pad_len)
    rft = np.apply_along_axis(rfft, 1, w2d_padded)
    
    return rft

   
if __name__ == "__main__":
    # check that r_window_to_matrix_eig works
    rft = r_window_to_matrix_eig(SINC)
    import matplotlib.pyplot as plt 
    plt.imshow(np.abs(rft.T), aspect="auto")
    plt.colorbar()
    plt.show()








