"""
Created on 2021.06.03 16:08:57

Author : Stephen Fay
"""

from scipy.fft import fft,ifft,fftshift,ifftshift

def window_to_box(window):
    return fftshift(fft(fftshift(window))) # gets taller

def box_to_window(box):
    return ifftshift(ifft(ifftshift(box))) # gets shorter
