"""
We would like to solve for $x$
$$w \ast x = d \Rightarrow Fx = Fd / Fw \Rightarrow x = F^{-1}\left\{ Fd/Fw \right\}$$
by our data is noisy

$$x' = F^{-1}\left\{ Fd/Fw \right\} + F^{-1}\left\{ Fn/Fw \right\}$$

if the noise has $\sigma=1$, we can get an idea of how bad the noise will be if we ddd

"""
print("\nINFO: Running conv_kernels.py")


# Local import
import sys
sys.path.append("..")
import helper as h
# Libraries
import numpy as np
from numpy.fft import irfft
import matplotlib.pyplot as plt

# set colorscheme
colors=plt.get_cmap('Set2').colors # get list of RGB color values
c0=colors[0]
c1=colors[1]


# `ntap` is the number of taps
# `lblock` is `2*nchan` which is the number of channels before transform
ntap, lblock = 4, 2048

# A sinc array
sinc = np.sinc(np.linspace(-ntap/2, ntap/2, ntap*lblock))
# Generate the matrix of eigenvalues
mat_eig = h.r_window_to_matrix_eig(sinc * np.hanning(ntap*lblock), 
                            ntap, lblock, zero_pad_len=72) # pad len must be even for result


# Wiener filter
def wien(Fw:np.ndarray):
    phi=0.07 # fixed threshold, (majic numbers bad ik)
    return abs(Fw)**2 * (1+phi)**2 / (abs(Fw)**2 + phi**2)

def get_kernel(col: int, trunkate: int=-1, is_wiener=False):
    """Get kernel coresponding to column col. Optionally trunkate to size (trunkate>0)"""
    if is_wiener is True:
        wienarr = wien(mat_eig[col,:])
        ker = irfft(wienarr/mat_eig[col,:])[::-1]
    elif is_wiener is False:
        ker = irfft(1/mat_eig[col,:])[::-1]
    else:
        raise Exception("Logic error. This should never execute.")
    if trunkate>0: 
        offset=2 # keep this close to zero, shifts center of array
        ker = np.concatenate((ker[(-trunkate)//2+offset:] , ker[:trunkate//2+offset]))
        x = np.arange((-trunkate//2)+offset,trunkate//2+offset)
    else:
        lenker = len(ker)
        ker = np.roll(ker,lenker//2)# reverse order and center in middle for plotting
        x = np.arange((-lenker)//2,lenker//2)
    return x,ker

def plot_six_panel(is_wiener=False):
    if is_wiener is True:
        suptitle="Wiener Filtered Kernels"
        savename = 'img/kernels_wiener_filtered.png'
    elif is_wiener is False:
        suptitle="Unfiltered Kernels"
        savename = 'img/kernels_unfiltered.png'
    else:
        raise Exception("Logic error. is_wiener must be boolean.")
    col=1023
    x,ker = get_kernel(col,is_wiener=is_wiener)
    fig,ax=plt.subplots(2,4,figsize=(13.5,4))
    ax[0,0].set_title(f"Time domain kernel\nColumn #{col} of {lblock}")
    ax[0,0].plot(x, ker, ".-", color=c0, markersize=0.7, linewidth=0.5)
    
    col=1000
    x,ker = get_kernel(col,is_wiener=is_wiener)
    ax[1,0].set_title(f"Time domain kernel\nColumn #{col} of {lblock}")
    ax[1,0].plot(x, ker, ".-", color=c0, markersize=0.7, linewidth=0.5)
    
    col=800
    x,ker = get_kernel(col,is_wiener=is_wiener)
    ax[0,1].set_title(f"Time domain kernel\nColumn #{col} of {lblock}")
    ax[0,1].plot(x, ker, ".-", color=c0, markersize=0.7, linewidth=0.5)
    x,ker = get_kernel(col,trunkate=15,is_wiener=is_wiener)
    ax[1,1].set_title(f"Zoomed col #{col} of {lblock}")
    ax[1,1].plot(x, ker,".-",color=c1,markersize=1.0,linewidth=0.5)
    
    col=300
    x,ker = get_kernel(col,is_wiener=is_wiener)
    ax[0,2].set_title(f"Time domain kernel\nColumn #{col} of {lblock}")
    ax[0,2].plot(x,ker,".-",color=c0,markersize=0.7,linewidth=0.5)
    x,ker = get_kernel(col,trunkate=15,is_wiener=is_wiener)
    ax[1,2].set_title(f"Zoomed col #{col} of {lblock}")
    ax[1,2].plot(x,ker,".-",color=c1,markersize=1.0,linewidth=0.5)
    
    col=0
    x,ker = get_kernel(col)
    ax[0,3].set_title(f"Time domain kernel\nColumn #{col} of {lblock}")
    ax[0,3].plot(x,ker,".-",color=c0,markersize=0.7,linewidth=0.5)
    x,ker = get_kernel(col,trunkate=15)
    ax[1,3].set_title(f"Zoomed col #{col} of {lblock}")
    ax[1,3].plot(x,ker,".-",color=c1,markersize=1.0,linewidth=0.5)
    
    fig.suptitle(suptitle,fontsize=20)
    fig.tight_layout()
    plt.savefig(savename,dpi=400)
    plt.show(block=True)
    return 

plot_six_panel(is_wiener=False)
plot_six_panel(is_wiener=True)

