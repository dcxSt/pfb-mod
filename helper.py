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

#%% windows 

def sinc_window(ntap=NTAP,lblock=LBLOCK):
    # i don't like np.arange, it's vulnerable to failur, 
    # should be linspace, but this is how R.S. implemented it, and it'll give a (tiny bit) different array
    return np.sinc(np.arange(-ntap/2,ntap/2,1/lblock)) 

def sinc_hamming(ntap=NTAP,lblock=LBLOCK):
    return np.hamming(ntap*lblock) * sinc_window(ntap,lblock)

    