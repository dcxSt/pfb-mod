# Mohan's plot
import numpy as np
import matplotlib.pyplot as plt

NX=5000 # number of x times to sample

colors=plt.get_cmap('Set2').colors # get list of RGB color values

def sinc_win(P,N):
    M=N*P
    x = (np.arange(0,M)-M/2)/N
    return np.sinc(x)

def get_pow(n, taper):
    """Get the freq response of a taper'd DFT."""
    delnu = np.linspace(1,0,100)
    pos = np.zeros(NX)
    fcenter = 10
    deltaf = np.arange(-3,5)
    t = np.linspace(0,1,n)
    # For each frequency calculate the response
    for i, df in enumerate(deltaf):
        for j,dnu in enumerate(delnu):
            x=taper(n)*np.exp(2j*np.pi*(fcenter+dnu)*t) # tapered sine wave
            X=np.fft.fft(x)                             # FFT
            pos[i*100+j] = np.abs(X[fcenter+df])**2     # power
    return pos

def pfb(x,nframe,ntap,taper=np.hamming):
    if taper is None: taper=np.ones
    h = taper(nframe*ntap)      # taper function
    s = sinc_win(ntap,nframe)   # sinc function
    X = np.fft.fft(s*h*x)       # FFT
    X = X[::ntap].copy()        # decimate (could chop b4 FFT too)
    return X

def get_pow_pfb(nframe, ntap, taper=np.hanning):
    """Get the frequency response of a PFB."""
    delnu = np.linspace(1,0,100)
    pos = np.zeros(NX)
    fcenter = 10
    deltaf = np.arange(-3,5)
    M=nframe*ntap
    t = np.linspace(0,ntap,M)
    for i, df in enumerate(deltaf):
        for j,dnu in enumerate(delnu):
            x=np.exp(2j*np.pi*(fcenter+dnu)*t)      # sine wave
            X=pfb(x,nframe,ntap,taper)              # PFB with optional taper
            pos[i*100+j] = np.abs(X[fcenter+df])**2 # power
    return pos

# helper, normalize to decibel scale
def decibel(y):
    return 10*np.log10(y/y.max())


pos1=get_pow(1024,np.ones)
pos2=get_pow(1024,np.hanning)
pos3=get_pow_pfb(1024,8,taper=np.hanning)
pos4=get_pow_pfb(1024,4,taper=np.hanning)
pos5=get_pow_pfb(1024,4,taper=None)

print(f"Shapes: {pos1.shape}, {pos2.shape}, {pos3.shape}, {pos4.shape}")

xl=np.linspace(-4,4,NX)
plt.figure(0)
plt.title('PFB vs DFT filter-bank response in each band')
plt.plot(xl,decibel(pos1),"--",label='FFT, no window',color=colors[0])
plt.plot(xl,decibel(pos2),"-.",label='FFT, Hann window',color=colors[1])
#plt.plot(xl,decibel(pos3),"-",label='PFB, 8 Taps, Hann window',color=colors[2])
plt.plot(xl,decibel(pos5),"--",label='PFB, 4 Taps, no window',color=colors[2])
plt.plot(xl,decibel(pos4),"-",label='PFB, 4 Taps, Hann window',color=colors[3])
plt.legend(loc='lower left')
plt.ylabel('Relative power (dB)',fontsize=12)
plt.xlabel('Frequency offset (channel)',fontsize=12)
plt.ylim(-80,1)
plt.tight_layout()
plt.savefig("img/sidelobes2.png",dpi=450)
plt.show()




