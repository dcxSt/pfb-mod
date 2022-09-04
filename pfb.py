#!/usr/bin/python3.8
"""
Created on 2021.06.04

Author : Stephen Fay
"""

import numpy as np
import helper as h
from scipy.fft import rfft,irfft,fft,ifft


# forward pfb as implemented in Richard Shaw's notebook
def forward_pfb(timestream, nchan=1025, ntap=4, window=h.sinc_hanning):
    """Performs the Chime PFB on a timestream
    
    Parameters
    ----------
    timestream : np.array
        Timestream to process
    nchan : int
        Number of frequencies we want out (probably should be odd
        number because of Nyquist)
    ntaps : int
        Number of taps.

    Returns
    -------
    pfb : np.ndarray[:, nchan]
        Array of PFB frequencies.
    """
    # number of samples in a sub block
    lblock = 2*(nchan - 1)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock==int(nblock): nblock=int(nblock)
    else: raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum 
    spec = np.zeros((nblock,nchan), dtype=np.complex128)

    # window function
    w = window(ntap, lblock)

    def s(ts_sec):
        return np.sum(ts_sec.reshape(ntap,lblock),axis=0)


    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi*lblock:(bi+ntap)*lblock].copy()

        # perform a real DFT (with applied, chunked window)
        spec[bi] = rfft(s(ts_sec * w)) 

    return spec 

def add_gaussian_noise(signal,sigma_proportion=0.001):
    """Adds gausian noise with standard deviation of 
        sigma_proportion times the mean of the absolute value signal
            to signal
    The reason is that depending on how long nchan is we can't guarantee what scale
    signal will be on, because it's abs value increases with length like sqrt(n)
    because that's how the FFT inside our pfb works

    Params
    signal : np.ndarray -- usually a 2 dimensional array
    """
    sigma = sigma_proportion * np.mean(np.abs(signal))
    return signal + np.random.normal(0,sigma,size=signal.shape)

def quantize(signal,delta=0.1):
    """Quantizes signal in intervals of delta in both real and imaginary parts, seperately"""
    q = lambda signal:np.floor((signal + delta/2) / delta) * delta 
    return q(np.real(signal)) + 1.0j*q(np.imag(signal)) 

def quantize_8_bit(signal, delta=0.5):
    """8-bit Quantizes the signal in intervals of delta in both real and imaginary parts, 4 bits (=16 steps) for each componant, seperately"""
    q = lambda signal:np.floor((signal + delta) / delta) * delta - delta/2 # doesn't quantize to zero, only to non-zero values
    real_quantized = np.clip(q(np.real(signal)) , -8*delta + delta/2 , 8*delta - delta/2) 
    imag_quantized = 1.0j*np.clip(q(np.imag(signal)) , -8*delta + delta/2 , 8*delta - delta/2) 
    return real_quantized + imag_quantized 

def quantize_real(real_signal,delta=0.1):
    """Quantizes signal in intervals of delta."""
    return np.floor((real_signal + delta/2) / delta) * delta  

def quantize_4_bit_real(real_signal, delta=0.1):
    """4-bit Quantizes signal in 16 intervals of delta"""
    return np.clip(np.floor((real_signal + delta) / delta) * delta - delta/2 , -8*delta +delta/2 , 8*delta - delta/2 )

# helper method for inverse pfb incase you get infinite values, put them to 10**100
def behead_infinite_values(arr):
    idxs = np.where(arr==np.inf)[0]
    arr[idxs] = 10**100*np.ones(len(idxs))
    print(len(idxs))
    return # don't have to return anything because arrays arr is a pointer

def bump_up_zero_values(arr):
    idxs = np.where(arr==0.0)[0]
    # arr[idxs] = 10**(-100)*np.ones(len(idxs)) # this is overkill
    arr[idxs] = 0.1 # in practice this is at most a single value
    print(len(idxs))
    return # don't have to return anything because arrays arr is a pointer

# pseudoinverse pfb
def inverse_pfb(spec, nchan=1025, ntap=4, window=h.sinc_hanning, wiener_thresh=0.0):
    """Performs pseudo inverse pfb, assumes circulant boundary conditions

    Parameters
    ----------
    spec : numpy 2d array (pfb channelized output)
        spec.shape = (:,nchan), the first entry is the length of the timestream
    nchan : int 
        usually 1025, it's the number of output channels
    ntap : int
        usually 4, it's the number of taps
    window : function
        the windowing function to apply, usually a sinc_hanning
    wiener_thresh : float
        if 0.0, no wiener filter is applied, if greater than zero, a wiener filter is applied with specified threshold 


    Returns
    -------
    timestream : numpy 1d array
        the pseudo-inverse timestream
    """
    # number of samples in a block
    lblock = 2*(nchan - 1)

    # sw_ts is what is returnd by applying sw matrix  to the original timstream chunk 
    # each subarray is like [g(1+4k)w1+g(17+4k)w17, ... ,g(16+2k)w16,g(32+4k)w32] ... i think
    sw_ts = np.apply_along_axis(irfft,1,spec) # what the ts looks like after applying SW matrices
    win = window(ntap,lblock) # this should be sing.hanning NOT sinc.hamming, see args

    # # temporary fill a list, change to numpy zeros later 
    # timestream = [] # np.zeros((lblock,sw_ts.shape[1]*ntap)) # initialize reconstructed timestream
    nblocks = (sw_ts.shape[0] + ntap - 1)/ntap 
    if nblocks==int(nblocks):nblocks=int(nblocks)
    else: raise Exception("nblocks should be an integer!, Something went wrong")
    timestream = np.zeros((lblock,nblocks*ntap),dtype=np.complex128) 

    # for loop for now so as not to make it too confusing
    for idx,(v,wslice) in enumerate(zip(sw_ts.T , win.reshape(ntap,lblock).T)):
        wslice_pad = np.roll(np.concatenate([np.flip(wslice),np.zeros(len(v)+ntap-1-len(wslice))]),-3) # flip, pad with zeros, roll
        # see pfb writup inverting pfb section for why we flip and roll, it's to do with how convolutions are defined

        # we can use rfft and irfft instead of fft and ifft, haven't got round to this yet... not sure if it's worth for our purposes really 
        ft_wslice = fft(wslice_pad) # this is a column of the eigenmatrix
        
        # make sure there are no zero value, because we are inverting it
        filt = np.ones(len(ft_wslice))  
        if wiener_thresh > 0.0:  
            filt = np.abs(ft_wslice)**2 / (wiener_thresh**2 + np.abs(ft_wslice)**2) * (1 + wiener_thresh**2)
            
        # if there are true zeros in the window slice, make them non zero to avoid divide by zero error, 
        # there shouldn't be any true zeros from experiance, but it's theoretically possible that there is one, however unlikely
        if 0.0 in ft_wslice:
            print("\nin inverse pfb: {} 0.0 values(s) encountered\n".format(len(np.where(ft_wslice == 0.0))))
            bump_up_zero_values(ft_wslice) # get rid of any zeros, replace with 10**(-100)
        ft_wslice_recip = 1/ft_wslice
        

        gtilde = ifft(fft(np.concatenate((v,v[:ntap-1])))*filt*ft_wslice_recip) # might be impossible so use rfft and irfft because not 'exactly' real, could use rfft on np.real(thing)
        # timestream.append(gtilde)# timestream[idx] = gtilde # old
        timestream[idx] = gtilde 


    # print("shape, stimstream :",np.array(timestream).shape)
    # input("input:")
    # print("timestream shape final ",timestream.shape)
    # timestream = np.array(timestream).T.flatten() # old

    return timestream.T.flatten()   # return the reconstructed timestream
                                    # the timestream will be returnd with dtype=complex but all the imaginary componants should be 0

