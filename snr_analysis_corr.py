"""
This script uses correlated SNRs to compare methods for filtering re-channelized quantized PFB'd data.

Let x_n be an i.i.d. sequence-realization of the random variable X. The mean 
<x_n> approximates the expected value <X>. The standard deviation of the
sequence approximates the std of the R.V. 

Let S be a R.V. that is the sky signal. Let N be a noise R.V. 
Let S1 and S2 be two sequences which are the sum of the same sky-signal
s, and two different noise signals N1, N2: 
    S1 = S + N1
    S2 = S + N2
So S1 and S2 are two time series. We use a PFB to coarsly channelize 
S1 and S2, we quantize the output to a few bits. Then we up-channelize
and clean with a filter or optimization method (nothing, Wiener filt, CG). 

Then we correlate. Because S and N are 
"""

import numpy as np
from numpy.fft import rfft,irfft

import pfb
from conjugate_gradient_stripped import conj_grad_with_prior

from datetime import datetime as dt
import pickle
import time
import argparse

# Parser args, first and only arg is the pseudo random seed
parser = argparse.ArgumentParser(description="Read first argument as int for pseudo random number generator seed.")
parser.add_argument("prng_seed", type=int, help="The first argument (integer)")
parser.add_argument("n_epochs", type=int, help="The number of epochs to compute (integer)")
args = parser.parse_args()
PRNG_SEED = args.prng_seed
N_EPOCHS = args.n_epochs


# Constants
LEN_EPOCH_LOG2 = 25 # For debugjob 22, real job 25 or 1/8 of a second
LEN_EPOCH      = 1<<LEN_EPOCH_LOG2 # 1<<26  # 1<<28 samples ~1.07 second's worth of data at 250 MSPS
#LEN_EPOCH      = 1<<24 # 1<<26  # 1<<28 samples ~1.07 second's worth of data at 250 MSPS
DELTA_4BIT     = 0.353  # Optimal delta for 15-level quantization
NFRAME         = 2048   # 1<<11
NTAP           = 4      # 1<<2
UPCHAN_FACTOR  = 32     # 1<<5, higher frequency resolution for upchannelization
TD_NOISE_SIGMA = 10     # Amount of noise added to time-domain signal
WIENER_THRESH  = 0.1
DROP_NROWS     = 4 # 10 # Number of up-channelized rows to drop to make sure we don't count the transient; make this number even

CONSTS = {
    "N_EPOCHS":N_EPOCHS,
    "LEN_EPOCH":LEN_EPOCH,
    "DELTA_4BIT":DELTA_4BIT,
    "NFRAME":NFRAME,
    "NTAP":NTAP,
    "UPCHAN_FACTOR":UPCHAN_FACTOR,
    "TD_NOISE_SIGMA":TD_NOISE_SIGMA,
    "PRNG_SEED":PRNG_SEED,
    "WIENER_THRESH":WIENER_THRESH,
    "DROP_NROWS":DROP_NROWS
}

#print("Number of up-channelized rows you'll get:", (1<<int(.5+np.log2(LEN_EPOCH) - np.log2(NFRAME*UPCHAN_FACTOR))) - NTAP+1 - DROP_NROWS)


def get_rfft_spec(sig, nframe):
    spec = np.reshape(sig, (len(sig)//nframe,nframe))
    return rfft(spec, axis=1)

def rechannelize(sig, quantize=False, usepfb=True, isupchan=False, 
                upchan_factor=4, wiener_thresh=0.0, drop_nrows=0, 
                only_upchan=False):
    """Channelize and optionally quantize and/or up-channelize a signal. 
    
    Uses global constants (only capitalized variables). 

    Parameters
    ----------
    sig : np.ndarray
        1d numpy real, time-domain signal
    quantize : bool
        If true, quantize the signal to four bits real + four bits imaginary once channelized
    usepfb : bool
        If True, channelize with a PFB. If False, channelize with STFTs. 
    isupchan : bool
        If True, up-channelize the signal by a factor of upchan_factor
    upchan_factor : int
        If upchan is True, the signal will be upchannelized by a factor of 
        If it's not a power of two this method might break. Just make it a power of two. 
    wiener_thresh : float
        Wiener threshold. Set to 0 for no Wiener filtering. This is only relevant if 
        upchan is True. 
    only_upchan : bool
        If this is True, only apply the higher-frequency channelization. Over-rides 
        quantization and inverse.

    returns
    -------
    spec : np.ndarray
        Sepctrum of shape (nrows, nchan) 
    """
    if usepfb is True:
        channelize = lambda x: pfb.forward_pfb(x, nchan=NFRAME//2+1, ntap=NTAP)
        upchan     = lambda x: pfb.forward_pfb(x, nchan=(NFRAME * upchan_factor)//2+1, ntap=NTAP)
        inverse    = lambda x: pfb.inverse_pfb(x, nchan=NFRAME//2+1, ntap=NTAP, 
                                               wiener_thresh=wiener_thresh)
    else:
        channelize = lambda x: get_rfft_spec(x, NFRAME)
        upchan     = lambda x: get_rfft_spec(x, NFRAME * upchan_factor)
        inverse    = lambda x: irfft(x, axis=1).flatten()
    if only_upchan is True:
        spec = upchan(sig)
    else:
        spec = channelize(sig)
        if quantize is True:
            std_spec = (np.std(np.real(spec)) + np.std(np.imag(spec)))/2 # for normalization
            spec = pfb.quantize_8_bit(spec, delta=std_spec * DELTA_4BIT)
        if isupchan is True:
            spec = upchan(np.real(inverse(spec)))
    if DROP_NROWS//2 > 0:
        spec = spec[DROP_NROWS//2:-DROP_NROWS//2,:]
    return spec

#def get_snr_corr(sig1, sig2, **kwargs):
#    """Estimate the Signal to Noise Ratio in each channel by correlating sig1 with sig2.
#    
#    Uses global constants (only capitalized variables). 
#
#    Parameters
#    ----------
#    sig1 : np.ndarray
#        1d numpy real, time-domain signal with noise, to be correlated with sig2
#    sig2 : np.ndarray
#        1d numpy real, time-domain signal with noise, to be correlated with sig1
#    **kwargs : dict
#        Arguments passed to rechannelize()
#
#    returns
#    -------
#    snr : np.ndarray
#        Signal to noise ratio in each channel, in dB
#    """
#    spec1 = rechannelize(sig1, **kwargs)
#    spec2 = rechannelize(sig2, **kwargs)
#    corr = (spec1 * np.conj(spec2)).mean(axis=0)
#    autocorr1 = (abs(spec1)**2).mean(axis=0)
#    autocorr2 = (abs(spec2)**2).mean(axis=0)
#    s = np.real(corr)             # Signal
#    n = autocorr1 - np.real(corr) # Noise
#    n2= autocorr2 - np.real(corr)
#    # print("s",s)
#    # print("n",n)
#    snr = 10 * np.log10(s/n)      # Signal to Noise Ratio
#    info_string = "{}{}{}{}\nSNR = {:.2f}".format("PFB" if kwargs.get("usepfb",False) else "FFT",
#        ", quantized" if kwargs.get("quantize",False) else "",
#        f", upchannelized by {kwargs.get('upchan_factor','<default>')}" if kwargs.get("isupchan",False) else "",
#        f", Wienered at {kwargs.get('wiener_thresh',False)}" if kwargs.get("wiener_thresh",False) else "",
#        snr.mean())
#    else:
#        print(info_string)
#    return snr




time_total_start = time.time()

kwargs_wien = {"quantize":True, "usepfb":True, "isupchan":True, "upchan_factor":UPCHAN_FACTOR, "wiener_thresh":0.1}
kwargs_nofilt = {"quantize":True, "usepfb":True, "isupchan":True, "upchan_factor":UPCHAN_FACTOR, "wiener_thresh":0.0}
kwargs_upchan_time_domain_sig = {"usepfb":True, "upchan_factor":UPCHAN_FACTOR, "only_upchan":True}

corrmean_wien   = []
corrmean_nofilt = []
corrmean_fp     = []
corrmean_1perc  = []
corrmean_3perc  = []
corrmean_5perc  = []
corrmean_10perc = []
prng = np.random.Generator(np.random.PCG64(seed=PRNG_SEED))
for epoch in range(N_EPOCHS):
    timeA = time.time()
    # Make a pseudo-random signal
    print("Generating signal...", end=" ")
    signal = prng.normal(0,1,LEN_EPOCH)
    sig1 = signal + prng.normal(0,TD_NOISE_SIGMA,LEN_EPOCH) 
    sig2 = signal + prng.normal(0,TD_NOISE_SIGMA,LEN_EPOCH) 
    timeB = time.time()
    print(f"took {timeB-timeA:.3f} seconds")

    
    # Infinite precision
    spec1=rechannelize(sig1,**kwargs_upchan_time_domain_sig) # Just channelize original signal 
    spec2=rechannelize(sig2,**kwargs_upchan_time_domain_sig) # directly to higher resolution
    corrmean_fp.append(np.mean(spec1 * np.conj(spec2), axis=0))
    del spec1, spec2
    timeC = time.time()
    print(f"{epoch+1}/{N_EPOCHS} mean power FP precision {np.mean(np.real(corrmean_fp[-1])):.1f}\t({timeC-timeB:.1f} s)")
    
    # Wiener filtered
    spec1=rechannelize(sig1,**kwargs_wien)
    spec2=rechannelize(sig2,**kwargs_wien)
    corrmean_wien.append(np.mean(spec1 * np.conj(spec2), axis=0))
    del spec1, spec2
    timeD = time.time()
    print(f"{epoch+1}/{N_EPOCHS} mean power wiener filtered {np.mean(np.real(corrmean_wien[-1])):.1f}\t({timeD-timeC:.1f} s)")
    
    # No filter
    spec1=rechannelize(sig1,**kwargs_nofilt)
    spec2=rechannelize(sig2,**kwargs_nofilt)
    corrmean_nofilt.append(np.mean(spec1 * np.conj(spec2), axis=0))
    del spec1, spec2
    timeE = time.time()
    print(f"{epoch+1}/{N_EPOCHS} mean power no filter {np.mean(np.real(corrmean_nofilt[-1])):.1f}\t({timeE-timeD:.1f} s)")
    
    # Optimize with CG
    conj_kwargs = {
        "frac_prior":0.01, 
        "delta":DELTA_4BIT, 
        "k":LEN_EPOCH//NFRAME, 
        "lblock":NFRAME, 
        "wiener_thresh":WIENER_THRESH,
        "npersave":7
    }
    for corrmean_list,frac_prior,npersave in zip((corrmean_1perc, corrmean_3perc, corrmean_5perc, corrmean_10perc),(0.01, 0.03, 0.05, 0.1), (7, 5, 4, 3)):
        timeF = time.time()
        conj_kwargs["frac_prior"] = frac_prior
        conj_kwargs["npersave"]   = npersave
        _, sig1_cg = conj_grad_with_prior(x=sig1,**conj_kwargs)
        _, sig2_cg = conj_grad_with_prior(x=sig2,**conj_kwargs)
        spec1=rechannelize(sig1_cg,**kwargs_upchan_time_domain_sig)
        spec2=rechannelize(sig2_cg,**kwargs_upchan_time_domain_sig)
        corrmean_list.append(np.mean(spec1 * np.conj(spec2), axis=0))
        del _, spec1, spec2
        timeG = time.time()
        print(f"{epoch+1}/{N_EPOCHS} mean power CG {int(0.5+100*frac_prior)}% {np.mean(np.real(corrmean_list[-1])):.1f}\t({timeG-timeF:.1f} s)")
    del sig1, sig2


time_total_end = time.time()
time_total_elapsed = time_total_end - time_total_start
print(f"\nDone!\nTotal time elapsed: {int(time_total_elapsed/3600)} hours {int((time_total_elapsed%3600)/60)} minutes {time_total_elapsed%60:.3f} seconds")

# Pickle the data
dumpdict = {}
for method,corrmean,kwargs in zip(
    ("wien","nofilt","fp","cg_1perc","cg_3perc","cg_5perc","cg_10perc"),
    (corrmean_wien, corrmean_nofilt, corrmean_fp, corrmean_1perc, corrmean_3perc, corrmean_5perc, corrmean_10perc),
    (kwargs_wien, kwargs_nofilt, kwargs_upchan_time_domain_sig, kwargs_upchan_time_domain_sig, kwargs_upchan_time_domain_sig, kwargs_upchan_time_domain_sig, kwargs_upchan_time_domain_sig)):
    dumpdict[method] = {"kwargs": kwargs, "CONSTS": CONSTS, "corrmean": corrmean}
now = dt.now()
with open(f'./snrdata/seed_{PRNG_SEED}_nepoch_{N_EPOCHS}_log2lenepoch_{LEN_EPOCH_LOG2}_{now}_snr_measurement.pkl','wb') as f:
    pickle.dump(dumpdict, f)









