import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft,fft,fftshift
from constants import *

#%% some windows for experimentation

def sinc_window(ntap=NTAP,lblock=LBLOCK):
    """Sinc window function
    
    Parameters
    ----------
    ntap : integer
        Number of taps
    lblock : integer
        Length of bloc. (lblock = 2*nchan)
        
    Returns
    -------
    window : np.array[ntaps * lblock]
    """
    # Sampling locations of sinc function
    X = np.arange(-ntap/2.0,ntap/2.0,1.0/lblock)
    return np.sinc(X)

def sinc_custom(r,offset=0,ntap=NTAP,lblock=LBLOCK):
    X = np.linspace(-r/2.0+offset,r/2.0+offset,ntap*lblock)
    return np.sinc(X)

def wabble(r=np.pi/4,sigma=0.2,ntap=NTAP,lblock=LBLOCK):
    sine = np.sin(np.linspace(-r,r,ntap*lblock))
    gauss = gaussian(np.linspace(-1.,1.,ntap*lblock),0,sigma)
    return sine * gauss

#%% display eigenvalue image

def chop_win(w,ntap=4,lblock=2048):
    """Chop lblock bits of len ntap of window to get ready for DFT"""
    if ntap*lblock!=len(w):raise Exception("len window incompatible")
    return np.reshape(w,(ntap,lblock)).T

def zero_padding(w2d,n_zeros=1024):
    pad = np.zeros((len(w2d),n_zeros))
    return np.concatenate([w2d,pad],axis=1)

def image_eigenvalues(w,ntap=NTAP,lblock=LBLOCK,name=None):
    """Images the eigenvalues of a window function
    
    Parameters
    ----------
    ntap : integer
        Number of taps
    lblock : integer
        Length of bloc. (lblock = 2*nchan)
    w : np.array[ntap * lblock]
        Window
    name : string
        Name of the window, if passed the figure displaed will be saved in ./figures/
        
    Displays:
        eigenvalues corresponding to this window function.
        the window and it's ntap chunks (4 chunks)
        the DFT of the window (boxcar-like thing)
    """
    if ntap*lblock!=len(w):raise Exception("len window incompatible")
    w2d = chop_win(w,ntap,lblock)
    w2d_padded = zero_padding(w2d)
    ft = np.apply_along_axis(rfft,1,w2d_padded)
    ft_abs = np.abs(ft)

    print("rfft shape and timestream blocked shape",ft.shape,w2d_padded.shape) # sanity check
    plt.subplots(figsize=(16,11))
    
    
    # plot the window and it's four slices
    plt.subplot(2,2,1)
    if ntap==4:
        chopped = chop_win(w).T
        plt.plot(chopped[0], alpha=0.5, color="red", label="segment 1")
        plt.plot(chopped[1], alpha=0.5, color="blue", label="segment 2")
        plt.plot(chopped[2], alpha=0.5, color="green", label="segment 3")
        plt.plot(chopped[3], alpha=0.5, color="orange", label="segment 4")

        plt.plot(np.linspace(0,lblock,len(w)),w,"-k",label="full window")
    else:
        plt.plot(w,"-k",label="full window")
    
    if name:
        plt.title("window {}".format(name),fontsize=18)
    else:plt.title("window",fontsize=18)
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    
    # image plot
    plt.subplot(2,2,2)
    # put a cieling on values, so not too high...
    ft_abs_ceil = ft_abs.copy()
    count=0
    for i,arr in enumerate(ft_abs):
        for j,el in enumerate(arr):
            if el>1.2:
                ft_abs_ceil[i][j]=1.2
    print(count)
    plt.imshow(ft_abs_ceil.T,aspect="auto")
    plt.xlabel("sample number",fontsize=16)
    plt.ylabel("rfft abs",fontsize=16)
    plt.colorbar()
    if name:
        plt.title("PFB Eigenvalues\n{}".format(name),fontsize=18)
        plt.savefig("./figures/pfb-colorplot-{}-window.png".format(name))
        np.save("./figures/{}-window".format(name),w)
    else:
        plt.title("PFB Eigenvalues",fontsize=18)
        
    # plot the boxcar (fft)
    bc = fftshift(fft(fftshift(w))) # the boxcar transform
    plt.subplot(2,2,3)
    plt.plot(bc)
    plt.title("fft window",fontsize=18)
    
    plt.subplot(2,2,4)
    plt.title("fft window zoom",fontsize=18)
    plt.plot(bc[int(ntap*lblock/2-10):int(ntap*lblock/2+10)])
    
    plt.tight_layout()
    
    plt.show()
    return

#%% main if run
if __name__ == "__main__":
    image_eigenvalues(sinc_window())