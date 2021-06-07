import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft,fft,fftshift
from scipy.signal import gaussian 
from constants import *
import helper

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

def image_eigenvalues(w,ntap=NTAP,lblock=LBLOCK,name=None,show="all",ghost=None):
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
    show : string
        Determines what to plot:
            "all" - will show four subplots = window, eigenvalues + the boxcar plots
            "window-eigen" - will show only the first two plots
            "eigen" - will show only the eigenvalues

    Displays:
        eigenvalues corresponding to this window function.
        the window and it's ntap chunks (4 chunks)
        the DFT of the window (boxcar-like thing)
    """
    if ntap*lblock!=len(w):raise Exception("len window incompatible")
    if show not in ("all","window-eigen","eigen"): raise Exception("\n\n'show' parameter invalid, please choose one of ['all','window-eigen','eigen']\n\n")
    w2d = helper.chop_win(w,ntap,lblock)
    w2d_padded = helper.zero_padding(w2d)
    ft = np.apply_along_axis(rfft,1,w2d_padded)
    ft_abs = np.abs(ft)

    print("rfft shape and timestream blocked shape",ft.shape,w2d_padded.shape) # sanity check
    figsize_dic = {"all":(16,11),"window-eigen":(16,5.5),"eigen":(6,5)}  
    plt.subplots(figsize = figsize_dic[show]) 
    subplots_dic = {"all":(221,222,223,224),"window-eigen":(121,122),"eigen":(None,111)}
    
    
    # plot the window and it's four slices
    if show in ("all","window-eigen"):plt.subplot(subplots_dic[show][0])
    if ntap==4:
        chopped = helper.chop_win(w).T
        plt.plot(chopped[0], alpha=0.5, color="red", label="segment 1")
        plt.plot(chopped[1], alpha=0.5, color="blue", label="segment 2")
        plt.plot(chopped[2], alpha=0.5, color="green", label="segment 3")
        plt.plot(chopped[3], alpha=0.5, color="orange", label="segment 4")

        plt.plot(np.linspace(0,lblock,len(w)),w,"-k",label="full window")
        if type(ghost)==type(np.array([0])):plt.plot(np.linspace(0,lblock,len(ghost)),ghost,"-.",color="grey",alpha=0.7,label="ghost")
    else:
        plt.plot(w,"-k",label="full window")

    if name:
        plt.title("window {}".format(name),fontsize=18)
    else:plt.title("window",fontsize=18)
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    
    # image plot
    plt.subplot(subplots_dic[show][1])
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
       
    # plot the boxcar (fft)
    if show=="all":
        bc = fftshift(fft(fftshift(w))) # the boxcar transform
        plt.subplot(subplots_dic[show][2])
        plt.plot(bc)
        plt.title("fft window",fontsize=18)
        
        plt.subplot(subplots_dic[show][3])
        plt.title("fft window zoom",fontsize=18)
        plt.plot(bc[int(ntap*lblock/2-10):int(ntap*lblock/2+10)])
        if type(ghost)==np.array([0]):plt.plot(helper.window_to_box(ghost)[int(ntap*lblock/2-10):int(ntap*lblock/2+10)])
        
        plt.tight_layout()
    if name:
        plt.title("PFB Eigenvalues\n{}".format(name),fontsize=18)
        plt.savefig("./figures/{}.png".format(name))
        np.save("./figures/{}.npy".format(name),w)
    else:
        plt.title("PFB Eigenvalues",fontsize=18)
    
    plt.show()
    return

#%% main if run
if __name__ == "__main__":
    image_eigenvalues(SINC_HAMMING,show="all")