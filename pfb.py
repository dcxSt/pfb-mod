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

# helper method for inverse pfb incase you get infinite values, put them to 10**100
def behead_infinite_values(arr):
    idxs = np.where(arr==np.inf)[0]
    arr[idxs] = 10**100*np.ones(len(idxs))
    print(len(idxs))
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

    # # temporary fill a list, change to numpy zeros later 
    # timestream = [] # np.zeros((lblock,sw_ts.shape[1]*ntap)) # initialize reconstructed timestream
    nblocks = (sw_ts.shape[0] + ntap - 1)/ntap 
    if nblocks==int(nblocks):nblocks=int(nblocks)
    else: raise Exception("nblocks should be an integer!, Something went wrong")
    timestream = np.zeros((lblock,nblocks*ntap),dtype=np.complex128) 

    # for loop for now so as not to make it too confusing
    for idx,(v,wslice) in enumerate(zip(sw_ts.T , w.reshape(ntap,lblock).T)):
        wslice_pad = np.roll(np.concatenate([np.flip(wslice),np.zeros(len(v)+ntap-1-len(wslice))]),-3) # flip, pad with zeros, roll
        # see pfb writup inverting pfb section for why we flip and roll

        # I think we can possibly use rfft and irfft, do this after implementation...
        ft_wslice = fft(wslice_pad) 
        ft_wslice_recip = 1/ft_wslice

        # get rid of infinite values if there are any, normaly not
        if np.inf in ft_wslice_recip: 
            print("\nin inverse pfb: {} np.inf value(s) encountered\n".format(len(np.where(ft_wslice_recip == np.inf))))
            behead_infinite_values(ft_wslice_recip) # get rid of infinity values, put them to 10**100

        gtilde = ifft(fft(np.concatenate((v,v[:ntap-1])))*ft_wslice_recip) # might be impossible so use rfft and irfft because not 'exactly' real, could use rfft on np.real(thing)
        # timestream.append(gtilde)# timestream[idx] = gtilde # old
        timestream[idx] = gtilde 


    # print("shape, stimstream :",np.array(timestream).shape)
    # input("input:")
    # print("timestream shape final ",timestream.shape)
    # timestream = np.array(timestream).T.flatten() # old

    return timestream.T.flatten() # return the reconstructed timestream

if __name__ == "__main__": 
    # # generate a two-sine-waves timestream
    # ntime = 2**11
    # ta = np.linspace(0.0, ntime / 2048, ntime, endpoint=False) 
    # ts = np.sin(2*np.pi * ta * 122.0) + np.sin(2*np.pi * ta * 378.1 + 1.0) 
    # spec_pfb = forward_pfb(ts,17,ntap=4)
    # recovered_ts = inverse_pfb(spec_pfb,nchan=17,ntap=4) 
    # res = recovered_ts - ts

    # time the script, takes about 1min 30seconds to run when lblock,ntap,nblocks = 2048,5,30000
    from datetime import datetime as dt 
    start_time = dt.today()

    lblock,ntap,nblocks = 32,4,500
    nchan = int(lblock/2+1) # assumes lblock is EVEN!
    ts = np.random.normal(0,1,size=lblock*ntap*nblocks) # initiate the timestream as gaussian noise
    spec_pfb = forward_pfb(ts,nchan,ntap) # pass it through the polyphase filter bank
    quantization_noise = 0.005
    spec_pfb = add_gaussian_noise(spec_pfb,sigma_proportion=quantization_noise) # add noise to the signal to simulate quantization
    recovered_ts = inverse_pfb(spec_pfb,nchan,ntap) # pass the filtered signal through the inverse pfb
    res = recovered_ts - ts 
    np.save("./data/nchan{}_ntap{}_nblocks{}_recovered_ts".format(nchan,ntap,nblocks),recovered_ts)
    np.save("./data/nchan{}_ntap{}_nblocks{}_input_ts".format(nchan,ntap,nblocks),ts)

    end_time = dt.today()
    print("Runtime : "+str(end_time-start_time))

    # downsample to plot, otherwise too taxing on matplotlib
    downsample = True
    ds_factor = lblock*nblocks // 500000 # this number is essentially the number of points to plot
    if ds_factor > 0 and downsample==True:
        print("downsampling")
        res = res[::ds_factor]
        recovered_ts = recovered_ts[::ds_factor]
        ts = ts[::ds_factor]



    # plot the residuals and everything
    import matplotlib.pyplot as plt
    # Residuals
    plt.subplots(figsize=(12,8))
    plt.subplot(221)
    plt.title("Recovered Timestream\nCirculant Fourier Method",fontsize=20)
    plt.plot(np.real(recovered_ts),lw=0.3)
    plt.xlabel("time x sample_rate",fontsize=15)
    plt.ylabel("E field amplitude",fontsize=15)

    plt.subplot(222)
    plt.title("Gaussian Noise Timestream\nResiduals",fontsize=20)
    plt.plot(np.real(res),lw=0.4) 
    plt.xlabel("time x sample_rate",fontsize=15)
    plt.ylabel("E field amplitude",fontsize=15)

    plt.subplot(223)
    plt.title("rfft",fontsize=20)
    plt.plot(np.abs(np.fft.rfft(np.real(recovered_ts))),"-",lw=0.3,label="abs recovered timstream")
    plt.plot(np.abs(np.fft.rfft(ts)),"-",lw=0.3,alpha=0.4,label="abs original timstream")
    plt.xlabel("frequency",fontsize=15)
    plt.ylabel("rfft amplitude",fontsize=15)
    plt.legend(loc="upper left")

    plt.subplot(224)
    if quantization_noise:plt.title("rfft Residuals\nQuantization noise {} percent".format(round(quantization_noise*100,1)),fontsize=20)
    else:plt.title("rfft Residuals",fontsize=20)
    plt.plot(np.abs(np.fft.rfft(np.real(res))),"k-",alpha=0.8,lw=0.7,label="absolute value")
    plt.plot(np.real(np.fft.rfft(np.real(res))),"g--",alpha=0.5,lw=0.7,label="real")
    plt.plot(np.imag(np.fft.rfft(np.real(res))),"--",color="orange",alpha=0.5,lw=0.7,label="imaginary")
    plt.xlabel("frequency",fontsize=15)
    plt.ylabel("rfft amplitude",fontsize=15)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("figures/recovered_timestream_residuals/gaussian_noise_nchan{}_ntap{}_nblocks{}_quantizationnoise{}percent.png".format(nchan,ntap,nblocks,np.round(quantization_noise*100,1)))
    plt.show()
    