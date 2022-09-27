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

def quantize_real(signal,delta=0.1):
    """Quantize signal with interval delta, assume real valued signal"""
    return np.floor((signal + delta/2) / delta) * delta

def quantize(signal,delta=0.1):
    """Quantizes signal in intervals of delta in both real and imaginary parts"""
    return quantize_real(np.real(signal),delta) + 1.0j*quantize_real(np.imag(signal),delta) 

def quantize_8_bit(signal, delta=0.5):
    """8-bit Quantizes the signal in intervals of delta in both real and imaginary parts, 4 bits (=15 intervals) for each componant, seperately"""
    q = lambda signal:np.floor((signal + delta) / delta) * delta - delta/2 # doesn't quantize to zero, only to non-zero values
    real_quantized = np.clip(quantize_real(np.real(signal), delta),
                             -8*delta + delta/2, 
                             8*delta - delta/2) 
    imag_quantized = 1.0j*np.clip(quantize_real(np.imag(signal), delta), 
                                  -8*delta + delta/2, 
                                  8*delta  - delta/2)
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
def inverse_pfb(spec, nchan = 1025, ntap = 4, 
        window = h.sinc_hanning, wiener_thresh = 0.0):
    """Performs pseudo inverse pfb, assumes circulant boundary conditions

    Parameters
    ----------
    spec : numpy 2d array (pfb channelized output)
        spec.shape = (:,nchan), the first entry is the length of the timestream
    nchan : int 
        Usually 1025, it's the number of output channels
    ntap : int
        Usually 4, it's the number of taps
    window : function
        The windowing function to apply, usually a sinc_hanning
    wiener_thresh : float
        If 0.0 is passed, no wiener filter is applied, if greater than 
        zero, a wiener filter is applied with specified threshold. The 
        wiener threshold should be the same as the standard deviation 
        of the noise?

    Returns
    -------
    np.ndarray
        The pseudo-inverse timestream. A 1d array. 
    """
    # Checks
    if spec.shape[1] != nchan: 
        raise Exception("spec.shape {spec.shape} != nchan {nchan}")
    # number of samples in a block
    lblock = 2*(nchan - 1)

    # If we represent the PFB as the successive application of three 
    # linear oprators, FSW, where F is a fourier transform, then 
    # sw_ts is what is returnd by applying sw matrix to the original 
    # timestream chunk. The k'th column/subarray will look like this.
    # 
    # [ g[k]*w[k]    + g[b+k]*w[b+k]  + g[2b+k]*w[2b+k] + g[3b+k]*w[3b+k], 
    #   g[b+k]*w[k]  + g[2b+k]*w[b+k] + g[3b+k]*w[2b+k] + g[4b+k]*w[3b+k],
    #   g[2b+k]*w[k] + g[3b+k]*w[b+k] + g[4b+k]*w[2b+k] + g[5b+k]*w[3b+k],
    #   g[3b+k]*w[k] + g[4b+k]*w[b+k] + g[5b+k]*w[2b+k] + g[6b+k]*w[3b+k],
    #   g[4b+k]*w[k] + g[5b+k]*w[b+k] + g[6b+k]*w[2b+k] + g[7b+k]*w[3b+k],
    #   g[5b+k]*w[k] + g[6b+k]*w[b+k] + g[7b+k]*w[2b+k] + g[8b+k]*w[3b+k],
    #   g[6b+k]*w[k] + g[7b+k]*w[b+k] + g[8b+k]*w[2b+k] + g[9b+k]*w[3b+k],
    #   ...
    # ]
    # 
    # Where ntap = 4, and b is lblock
    sw_ts = np.apply_along_axis(irfft, 1, spec) 
    # win is the actual array
    win = window(ntap, lblock) # Should be sinc.hanning NOT sinc.hamming
    nblocks = (sw_ts.shape[0] + ntap - 1)/ntap 
    if nblocks == int(nblocks): nblocks = int(nblocks)
    else: raise Exception("nblocks should be an integer, terminating.")
    timestream = np.zeros((lblock, nblocks*ntap), dtype=np.complex128) 

    # Implement as for loop for now so as not to make it too confusing
    # but in theory we could use ndarrays and transposes
    for idx,(v,wslice) in enumerate(zip(sw_ts.T , win.reshape(ntap,lblock).T)):
        # This next line looks like some disgusting python ninja move, 
        # but it's the cleanest way I could come up with, and it's very 
        # compact 
        wslice_pad = np.roll(
                np.concatenate(
                    [
                        np.flip(wslice), 
                        np.zeros(len(v)+ntap-1-len(wslice))
                    ]),
                -3) # flip, pad with zeros, roll
        # See pfb writup--inverting pfb section for why we flip and 
        # roll, it's to do with getting the spectrum to allign with the
        # multiplying array and how convolutions are defined.

        # We can use rfft and irfft instead of fft and ifft, haven't got round to this yet... not sure if it's worth for our purposes really 
        ft_wslice = fft(wslice_pad) # this is a column of the eigenmatrix
        
        # Make sure there are no zero values, because we are inverting it
        filt = np.ones(len(ft_wslice))  
        if wiener_thresh > 0.0:  
            filt = np.abs(ft_wslice)**2 / (wiener_thresh**2 + np.abs(ft_wslice)**2) * (1 + wiener_thresh**2)
        # If there are true zeros in the window slice, make them non 
        # zero to avoid divide by zero error, there shouldn't be any 
        # true zeros from experiance, but it's theoretically possible 
        # that there is one, however unlikely.
        if 0.0 in ft_wslice:
            print("\nWARNING: inverse pfb {} 0.0 values(s) encountered\n".format(len(np.where(ft_wslice == 0.0))))
            bump_up_zero_values(ft_wslice) # replace zeros with 10^-100
        ft_wslice_recip = 1/ft_wslice
        
        # Below, might be impossible so use rfft and irfft because not 
        # 'exactly' real, could use rfft on np.real(thing)
        gtilde = ifft(fft(np.concatenate((v,v[:ntap-1]))) * filt * ft_wslice_recip) 
        timestream[idx] = gtilde 


    # print("shape, stimstream :",np.array(timestream).shape)
    # input("input:")
    # print("timestream shape final ",timestream.shape)
    # timestream = np.array(timestream).T.flatten() # old

    # Return the reconstructed timestream. The timestream is returned 
    # as a numpy ndarray with dtype=complex, but all the imaginary 
    # componants will arise due to edge effects and quantization errors.
    # In other words, in essence, it is a real array. 
    return timestream.T.flatten()   
                                    


# # Line for line copy of implementation in Jon's note
# # Why is it hamming here, shouldn't it be hanning? I keep forgetting 
# # which one is the one that's actually implemented.
# # What is the pfb.sinc_hamming function???
# def inverse_pfb_fft_filt(dat, ntap=4, window=pfb.sinc_hamming, 
#         thresh = 0.0):
#     dd  = irfft(dat, axis=1)
#     win = window(ntap, dd.shape[1])
#     win = np.reshape(win, [ntap, len(win)//ntap])
#     mat = np.zeros(dd.shape, dtype=dd.dtype)
#     mat[:ntap,:] = win
#     matft = rfft(mat, axis=0)
#     ddft  = rfft(dd, axis=0)
#     if thresh > 0:
#         filt = np.abs(matft)**2/(thresh**2+np.abs(matft)**2)*(1+thresh**2)
#         ddft = ddft*filt
#     return irfft(ddft/np.conj(matft), axis=0)


### TESTS

if __name__=="__main__":
    from matplotlib import pyplot as plt
    print("INFO: Running tests.")
    print("INFO: Visual Tests.")
    print("INFO: Test quantize_real(), expect to see quantized signal")

    print("INFO: Test quantize(), see real and imaginary")

    print("INFO: Test quantize_8_bit(), count 15 levels of quantization in each")







