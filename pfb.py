#!/usr/bin/python3.8
"""
Created on 2021.06.04

Author : Stephen Fay
"""

from constants import * # in particular imports NTAP = 4 and LBLOCK = 2048
from helper import sinc_hamming
from scipy.fft import rfft,irfft,fft,ifft

# forward pfb as implemented in Richard Shaw's notebook
def forward_pfb(timestream,nchan=NCHAN,ntap=NTAP,window=sinc_hamming):
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
    w = window(ntap,lblock)

    def s(ts_sec):
        return np.sum(ts_sec.reshape(ntap,lblock),axis=0)


    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi*lblock:(bi+ntap)*lblock].copy()

        # perform a real DFT (with applied, chunked window)
        spec[bi] = rfft(s(ts_sec * w)) 

    return spec 

# helper method for inverse pfb incase you get infinite values, put them to 10**100
def behead_infinite_values(arr):
    idxs = np.where(arr==np.inf)[0]
    arr[idxs] = 10**100*np.ones(len(idxs))
    return # don't have to return anything because arrays arr is a pointer

# pseudoinverse pfb
def inverse_pfb(spec,nchan=NCHAN,ntap=NTAP,window=sinc_hamming):
    """Performs pseudo inverse pfb, assumes circulant boundary conditions

    Parameters
    ----------
    pfb_spec : numpy 2d array
        spec.shape = (:,nchan), the first entry is the length of the timestream

    Returns
    -------
    timestream : numpy 1d array
        the pseudo-inverse timestream
    """
    # number of samples in a block
    lblock = 2*(nchan - 1)

   # sw_ts is what is returnd by applying sw matrix  to the original timstream chunk 
   # each subarray is like [g(1+4k)w1+g(17+4k)w17, ... ,g(16+2k)w16,g(32+4k)w32] ... i think
    sw_ts = np.apply_along_axis(irfft,1,spec) 
    w = window(ntap,lblock) 

    # temporary fill a list, change to numpy zeros later 
    timestream = [] # np.zeros((lblock,sw_ts.shape[1]*ntap)) # initialize reconstructed timestream
    # print("timstream shape init ",timestream.shape)

    # for loop for now so as not to make it too confusing
    for idx,(v,wslice) in enumerate(zip(sw_ts.T , w.reshape(ntap,lblock).T)):
        wslice_pad = np.roll(np.concatenate([np.flip(wslice),np.zeros(len(v)+ntap-1-len(wslice))]),-3) # flip, pad with zeros, roll
        # see pfb writup inverting pfb section for why we flip and roll

        # I think we can possibly use rfft and irfft, do this after implementation...
        ft_wslice = fft(wslice_pad) 
        ft_wslice_recip = 1/ft_wslice

        # get rid of infinite values
        if np.inf in ft_wslice_recip: 
            print("\nin inverse pfb: {} np.inf value(s) encountered\n".format(len(np.where(ft_wslice_recip == np.inf))))
            behead_infinite_values(ft_wslice_recip) # get rid of infinity values, put them to 10**100

        gtilde = ifft(fft(np.concatenate((v,v[:ntap-1])))*ft_wslice_recip) # might be impossible so use rfft and irfft because not 'exactly' real, could use rfft on np.real(thing)
        timestream.append(gtilde)# timestream[idx] = gtilde 

    # print("timestream shape final ",timestream.shape)
    timestream = np.array(timestream).T.flatten() 

    return timestream

if __name__ == "__main__": 
    # timestream = np.random.normal(size=LBLOCK*NTAP*10)
    # generate a two-sine-waves timestream
    ntime = 2**11
    ta = np.linspace(0.0, ntime / 2048, ntime, endpoint=False) 
    ts = np.sin(2*np.pi * ta * 122.0) + np.sin(2*np.pi * ta * 378.1 + 1.0) 
    spec_pfb = forward_pfb(ts,17,ntap=4)
    recovered_ts = inverse_pfb(spec_pfb,nchan=17,ntap=4)
    res = recovered_ts - ts

    # plot the residuals and everything
    import matplotlib.pyplot as plt
    # Residuals
    plt.subplots(figsize=(12,8))
    plt.subplot(221)
    plt.title("Recovered Timestream\nCirculant Fourier Method",fontsize=20)
    plt.plot(np.real(recovered_ts),color="black",lw=0.8)
    plt.xlabel("time x sample_rate",fontsize=15)
    plt.ylabel("E field amplitude",fontsize=15)

    plt.subplot(222)
    plt.title("Residuals",fontsize=20)
    plt.plot(np.real(res),color="black",lw=0.8) 
    plt.xlabel("time x sample_rate",fontsize=15)
    plt.ylabel("E field amplitude",fontsize=15)

    plt.subplot(223)
    plt.title("rfft",fontsize=20)
    plt.plot(np.abs(np.fft.rfft(np.real(recovered_ts))),label="recovered timstream")
    plt.plot(np.abs(np.fft.rfft(ts)),alpha=0.4,label="original timstream")
    plt.xlabel("frequency",fontsize=15)
    plt.ylabel("rfft amplitude",fontsize=15)
    plt.legend()

    plt.subplot(224)
    plt.title("rfft Residuals",fontsize=20)
    plt.plot(np.abs(np.fft.rfft(np.real(res))),"k-",alpha=0.4,label="absolute value")
    plt.plot(np.real(np.fft.rfft(np.real(res))),"g-",alpha=0.6,label="real")
    plt.plot(np.imag(np.fft.rfft(np.real(res))),color="orange",alpha=0.6,label="imaginary")
    plt.xlabel("frequency",fontsize=15)
    plt.ylabel("rfft amplitude",fontsize=15)
    plt.legend()

    plt.tight_layout()
    # plt.savefig("figures/recovered_timestream_circulant_fourier_method_residuals")
    plt.show()
    