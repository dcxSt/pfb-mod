import jax.numpy as np # still gotta test to see if old method work with this 
import numpy 
import matplotlib.pyplot as plt
from scipy.fft import rfft,fft,fftshift
from scipy.signal import gaussian 
from constants import *
import helper as h

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
    w2d = h.chop_win(w,ntap,lblock)
    w2d_padded = h.zero_padding(w2d)
    ft = np.apply_along_axis(rfft,1,w2d_padded)
    ft_abs = np.abs(ft)

    print("rfft shape and timestream blocked shape",ft.shape,w2d_padded.shape) # sanity check
    figsize_dic = {"all":(16,11),"window-eigen":(16,5.5),"eigen":(6,5)}  
    plt.subplots(figsize = figsize_dic[show]) 
    subplots_dic = {"all":(221,222,223,224),"window-eigen":(121,122),"eigen":(None,111)}
    
    
    # plot the window and it's four slices
    if show in ("all","window-eigen"):plt.subplot(subplots_dic[show][0])
    if ntap==4:
        chopped = h.chop_win(w).T
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
        if type(ghost)==np.array([0]):plt.plot(h.window_to_box(ghost)[int(ntap*lblock/2-10):int(ntap*lblock/2+10)])
        
        plt.tight_layout()
    if name:
        plt.title("PFB Eigenvalues\n{}".format(name),fontsize=18)
        plt.savefig("./figures/{}.png".format(name))
        np.save("./figures/{}.npy".format(name),w)
    else:
        plt.title("PFB Eigenvalues",fontsize=18)
    
    plt.show()
    return

def image_eig2(window,save_fig=False):
    """Images the eigenvalues of a window function
    
    Parameters
    ----------
    window : np.array[ntap * lblock] (assumes 4*2048)
        Window
    save_fig : boolean
        if true will save figure and window array with datetime tag
    """
    from datetime import datetime as dt
    strdatetime = dt.today().strftime("%Y-%m-%d_%H.%M.%S")

    ### Loss and reward functions

    mat_eig = h.r_window_to_matrix_eig(window)
    thresh_025 = np.count_nonzero(np.abs(mat_eig)<0.25)
    thresh_001 = np.count_nonzero(np.abs(mat_eig)<0.1)

    # ### modified spectrum
    plt.subplots(figsize=(16,10))

    ### window

    plt.subplot(221)
    plt.plot(abs(window),"k-.",alpha=0.3,label="abs")
    plt.plot(np.imag(window),alpha=0.4,color="orange",label="imaginary")
    plt.plot(SINC,color="grey",alpha=0.6,label="sinc")
    plt.plot(window,"b-",label="real")
    plt.title("Window\n{}".format(strdatetime),fontsize=10)
    plt.legend()

    ### eig plot

    plt.subplot(222)
    rft = h.r_window_to_matrix_eig(window).T
    rft = numpy.array(rft)
    rft[0][0]=0.0 # make one of them zero to adjust the scale of the plot
    plt.imshow(np.abs(rft),cmap="gist_ncar",aspect="auto")
    # plt.title("Eigenvalues\nLoss Eig : {}(0.207)\nThresh 0.25 : {} (9519)\nThresh 0.1 : {} (1529)".format(round(0.0,3),thresh_025,thresh_001),fontsize=20)
    plt.title("Eigenvalues\nThresh 0.25 : {} (9519) --> {:.2f}%\nThresh 0.1 : {} (1529) --> {:.2f}%".format(thresh_025,100*thresh_025/9519,thresh_001,100*thresh_001/1529),fontsize=10)
    # in above line 0.0 should be l_eig
    plt.colorbar()

    ### box

    box = h.window_pad_to_box(window,10.0)
    short_box = box[int(len(box)/2):int(len(box)/2+750)]
    # scale = max(np.abs(short_box)) # this is the scale of the fitler, determines where we put lines

    box_sinc = h.window_pad_to_box(SINC,10.0)
    short_box_sinc = box_sinc[int(len(box_sinc)/2):int(len(box_sinc)/2+750)]
    scale = max(np.abs(short_box_sinc)) # now we can scale everyone down to where to peak in logplot is zero
    box,short_box,box_sinc,short_box_sinc = box/scale,short_box/scale,box_sinc/scale,short_box_sinc/scale

    # metrics for evaluating the box, thicknesses of the box at different scales
    th2,th3,th4,th5,th6 = h.metric_sidelobe_thicknesses(window)

    ### plot the box

    plt.subplot(223)
    plt.semilogy(np.abs(short_box_sinc),"b-",alpha=0.7,label="sinc")
    plt.semilogy(np.abs(short_box),"k-",alpha=0.7,label="window")
    # plt.title("log Box zoom\nWidth Loss : {} \tHeight Loss : {}".format(round(l_width,3),round(l_height,3)),fontsize=20)
    plt.title("log Box zoom (th_x = boxcar thickness at 10^-x)\nth_2 = {:.2f}%   th_3 = {:.2f}%\nbaseline: th_2 = 0.08%   th_3 = 0.40% :baseline".format(th2,th3),fontsize=10)
    plt.grid(which="both")
    plt.legend()


    plt.subplot(224)
    plt.semilogy(np.abs(box_sinc),"b-",alpha=0.5,label="sinc")
    plt.semilogy(np.abs(box),"k-",alpha=0.5,label="window")
    plt.semilogy(np.ones(len(box))*10**(-5),color="green",alpha=0.5,label="10^-5")
    plt.semilogy(np.ones(len(box))*10**(-6),color="green",alpha=0.5,label="10^-6")
    plt.title("log Box\nth_4 = {:.2f}%   th_5 = {:.2f}%    th_6 = {:.2f}%\nth_4 = 1.40%   th_5 = 4.64%   th_6 = 14.82%".format(th4,th5,th6),fontsize=10)
    # plt.title("log Box",fontsize=20)
    plt.grid(which="both")
    plt.legend()

    plt.tight_layout()

    if save_fig==True:
        np.save("figures/experiments/series6_{}.npy".format(strdatetime),window)
        print("saved window")
        plt.savefig("figures/experiments/series6_{}.png".format(strdatetime))
        print("saved figure")

    plt.show()


#%% main if run
if __name__ == "__main__":
    image_eigenvalues(SINC_HAMMING,show="all")