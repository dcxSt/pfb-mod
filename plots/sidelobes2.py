# Mohan's plot
import numpy as np
import matplotlib.pyplot as plt

NX=5000 # number of x times to sample

colors=plt.get_cmap('Set2').colors # get list of RGB color values

def sinc_win(P,N):
    M=N*P
    x = (np.arange(0,M)-M/2)/N
    return np.sinc(x)

def get_pow(N, win):
    delnu = np.linspace(1,0,100)
    pos = np.zeros(NX)
    fcenter = 10
    deltaf = np.arange(-3,5)
    t = np.linspace(0,1,N)
    for i, df in enumerate(deltaf):
        for j,dnu in enumerate(delnu):
            x=win(N)*np.exp(2j*np.pi*(fcenter+dnu)*t)
            X=np.fft.fft(x)
#             print(f"dist k-nu is {df-dnu}")
            pos[i*100+j] = np.abs(X[fcenter+df])**2
#     print(pos)
            
    return pos

def pfb(x,N,ntap):
    M=N*ntap
    s = sinc_win(ntap,N)
    h = np.hanning(M)
    # equivalent
#     y = s*h*x
#     y = y.reshape(-1,N,order='c')
#     y = np.sum(y, axis=0)
#     X = np.fft.fft(y)
    X = np.fft.fft(s*h*x)
    X = X[::ntap].copy()
    return X

def get_pow_pfb(N, ntap):
    
    delnu = np.linspace(1,0,100)
    pos = np.zeros(NX)
    fcenter = 10
    deltaf = np.arange(-3,5)
    M=N*ntap
    # Convention: N is for 1 sec. basically N Hz. So n taps in ntap seconds.
    t = np.linspace(0,ntap,M)
    for i, df in enumerate(deltaf):
        for j,dnu in enumerate(delnu):
            x=np.exp(2j*np.pi*(fcenter+dnu)*t)
            X=pfb(x,N,ntap)
#             print(f"dist k-nu is {df-dnu}")
            pos[i*100+j] = np.abs(X[fcenter+df])**2
#     print(pos)
    return pos

# helper, normalize to decibel scale
def decibel(y):
    return 10*np.log10(y/y.max())


hann=lambda N: np.hanning(N)
flat=lambda N: np.ones(N)
pos1=get_pow(1024,flat)
pos2=get_pow(1024,hann)
pos3=get_pow_pfb(1024,8)
pos4=get_pow_pfb(2048,4)

xl=np.linspace(-4,4,NX)
plt.figure(0)
plt.title('8 tap Hanning-PFB leakage comparison')
plt.plot(xl,decibel(pos1),"--",label='No window',color=colors[0])
plt.plot(xl,decibel(pos2),"-.",label='Hann only',color=colors[1])
plt.plot(xl,decibel(pos3),label='Hann-PFB 8 Taps',color=colors[2])
plt.plot(xl,decibel(pos4),label='Hann-PFB 4 Taps',color=colors[3])
plt.legend()
plt.ylabel('Relative power (dB)',fontsize=12)
plt.xlabel('Frequency offset (channel)',fontsize=12)
plt.ylim(-80,1)

plt.tight_layout()
plt.savefig("img/sidelobes2.png",dpi=450)
plt.show()




