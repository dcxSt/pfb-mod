import numpy as np
import matplotlib.pyplot as plt

NX=5000 # global var, number of x times to sample
FREQ_BIN=100 # center frequency bin chosen arbitrarily
colors=plt.get_cmap('Set2').colors # get list of RGB color values

def sinc_win(ntap,nframe):
    n=nframe*ntap
    x = (np.arange(0,n)-n/2)/nframe
    return np.sinc(x)

def response_of_freq_x_in_bin_fft(x_freq, freq_bin, nframe, taper):
    time_arr = np.linspace(0, 1, nframe)
    tapered_sine_wave = taper(nframe) * np.exp(2j*np.pi*x_freq*time_arr)
    return np.abs(np.fft.rfft(tapered_sine_wave)[freq_bin])**2

def get_power_fft(nframe, taper=np.hanning):
    """Get the freq response of a tapered DFT"""
    freqs = np.linspace(-3.5,3.5,NX) + FREQ_BIN # unitless wave number
    responses = [response_of_freq_x_in_bin_fft(freq, FREQ_BIN, nframe, taper) for freq in freqs]
    return freqs, responses

def pfb(x,nframe,ntap,taper=np.hanning):
    if taper is None: taper=np.ones
    h = taper(nframe*ntap)      # taper function
    s = sinc_win(ntap,nframe)   # sinc function
    X = np.fft.fft(s*h*x)       # FFT
    X = X[::ntap].copy()        # decimate (could chop b4 FFT too)
    return X

def response_of_freq_x_in_bin(x_freq, freq_bin, nframe, ntap, taper):
    """x_freq and y_freq are wave numbers, so t is from 0 to 1"""
    time_arr = np.linspace(0,ntap,nframe*ntap)
    tapered_sine_wave = np.exp(2j*np.pi*x_freq*time_arr)
    return np.abs(pfb(tapered_sine_wave, nframe, ntap, taper)[freq_bin])**2

def get_power_pfb(nframe, ntap, taper=np.hanning):
    freqs = np.linspace(-3.5,3.5,NX) + FREQ_BIN # unitless wave number
    responses = [response_of_freq_x_in_bin(freq, FREQ_BIN, nframe, ntap, taper) for freq in freqs]
    return freqs,responses

# helper, normalize to decibel scale
def decibel(y):
    return 10*np.log10(y/y.max())


freqs,pos1=get_power_fft(4096,np.ones)
_,pos2=get_power_fft(4096,np.hanning)
_,pos3=get_power_pfb(4096,8,taper=np.hanning)
_,pos4=get_power_pfb(4096,4,taper=np.hanning)
_,pos5=get_power_pfb(4096,4,taper=np.ones)

pos1=np.array(pos1)
pos2=np.array(pos2)
pos3=np.array(pos3)
pos4=np.array(pos4)
pos5=np.array(pos5)

print(f"Shapes: {pos1.shape}, {pos2.shape}, {pos3.shape}, {pos4.shape}")

#xl=np.linspace(-4,4,NX) # depricated
plt.figure(0)
plt.title('PFB vs DFT filter-bank response in each band')
plt.plot(freqs-FREQ_BIN,decibel(pos1),"--",label='FFT, no window',color=colors[0])
plt.plot(freqs-FREQ_BIN,decibel(pos2),"-.",label='FFT, Hann window',color=colors[1])
plt.plot(freqs-FREQ_BIN,decibel(pos3),"-",label='PFB, 8 Taps, Hann window',color=colors[2])
plt.plot(freqs-FREQ_BIN,decibel(pos5),"--",label='PFB, 4 Taps, no window',color=colors[4])
plt.plot(freqs-FREQ_BIN,decibel(pos4),"-",label='PFB, 4 Taps, Hann window',color=colors[3])
plt.axvspan(-.5, .5, color='lightgrey', alpha=0.5)
#plt.axvline(.5,color='lightgrey',linestyle='--')
#plt.axvline(-.5,color='lightgrey',linestyle='-')
plt.legend(loc='lower left')
plt.ylabel('Relative power (dB)',fontsize=12)
plt.xlabel('Frequency offset (channel)',fontsize=12)
plt.ylim(-80,1)
plt.tight_layout()
plt.savefig("img/sidelobes2.png",dpi=450)
plt.show()


