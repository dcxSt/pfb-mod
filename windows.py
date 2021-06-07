#!/usr/bin/python3.8

"""
Created on 2021-06-07

Author : Stephen Fay
"""

import numpy as np
from constants import *
import helper as h

#%% spectrum transformers (spectrum ~ ft_block)
f3 = lambda x:x*(1/(np.abs(x)+0.1)+0.3)
f4 = lambda x:x*(1/(np.abs(x)+0.000001))
f5 = lambda x:x*(1/(np.abs(x)+0.01))
f6 = lambda x:x*(1/(np.abs(x)+0.000000000001))
f7 = lambda x:x*(1/(np.abs(x)+10.0**(-50)))

# repete the transformation procedure n times
def repete_func(f,ft_block,n,ntap=NTAP,lblock=LBLOCK):
    # apply f to ft_block n times
    for i in range(n):
        ft_block = f(ft_block) 
        complex_rec = h.matrix_eig_to_window_complex(ft_block,ntap)
        ft_block = h.window_to_matrix_eig(np.real(complex_rec),ntap,lblock)
    return ft_block,complex_rec

#%% candidate replacement windows
def william_wallace(ntap=NTAP,lblock=LBLOCK):
    """
    input : a sinc or sinc hamming window, produces similar results
    output : a candidate window that doesn't have as much leaking
    """
    sinc = h.sinc_window(ntap,lblock) 
    # input("type(sinc) {}\nsinc start: {}".format(type(sinc),sinc[:10]))
    ft_block = h.window_to_matrix_eig(sinc,ntap,lblock)
    ft_block,complex_rec = repete_func(f6,ft_block,10,ntap,lblock) # result is almost identitcal if we use f7 instead of f6
    candidate_1 = np.real(complex_rec) 
    return candidate_1

#%% windows to export
WILLIAM_WALLACE = william_wallace()

#%% run this file
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime as dt

    ntap,lblock = NTAP,32 # LBLOCK 
    sinc = h.sinc_window(ntap,lblock)

    ft_block_original = h.window_to_matrix_eig(sinc,ntap,lblock) # alternatively use SINC_HAMMING
    ft_block = ft_block_original.copy()

    ft_block,complex_rec = repete_func(f6,ft_block,10,ntap,lblock)
    abs_rec = np.abs(complex_rec)
    imag_rec = np.imag(complex_rec)
    reconstructed_window = np.real(complex_rec)

    ### modified spectrum

    plt.subplots(figsize=(16,14))
    plt.subplot(431)
    plt.imshow(np.real(ft_block_original),aspect="auto")
    plt.title("real original")
    plt.colorbar()

    plt.subplot(432)
    plt.imshow(np.abs(ft_block_original),aspect="auto")
    plt.title("absolute original")
    plt.colorbar()

    plt.subplot(433)
    plt.imshow(np.imag(ft_block_original),aspect="auto")
    plt.title("imaginary original")
    plt.colorbar()

    ### corresponding reconstruction from window

    plt.subplot(434)
    plt.imshow(np.real(ft_block),aspect="auto")
    plt.title("real (constructed from window)\nTHE ACTUAL THING")
    plt.colorbar()

    plt.subplot(435)
    plt.imshow(np.abs(ft_block),aspect="auto")
    plt.title("absolute (constructed from window)\nTHE ACTUAL THING")
    plt.colorbar()

    plt.subplot(436)
    plt.imshow(np.imag(ft_block),aspect="auto")
    plt.title("imaginary (constructed from window)\nTHE ACTUAL THING")
    plt.colorbar()

    ### the window

    plt.subplot(425)
    plt.plot(abs_rec,"k-.",alpha=0.3,label="abs")
    plt.plot(imag_rec,alpha=0.4,color="orange",label="imaginary")
    plt.plot(sinc,color="grey",alpha=0.4,label="sinc")
    plt.plot(reconstructed_window,"b-",label="real")
    plt.title("window")
    plt.legend()

    ### the boxcar 

    box = h.window_to_box(reconstructed_window)

    plt.subplot(426)
    short_box = box[int(ntap*lblock/2-15):int(ntap*lblock/2+15)]
    plt.plot(np.real(short_box),"b-",alpha=0.3,label="real")
    plt.plot(np.abs(short_box),"k-",label="abs")
    plt.grid()
    plt.title("box zoom")
    plt.legend()

    plt.subplot(427)
    short_box = box[int(ntap*lblock/2-150):int(ntap*lblock/2+150)]
    plt.plot(np.real(short_box),"b-",alpha=0.3,label="real")
    plt.plot(np.abs(short_box),"k-",label="abs")
    plt.title("box zoom")
    plt.grid()
    plt.legend()


    plt.subplot(428)
    plt.plot(np.real(box),"b-",alpha=0.3,label="real")
    plt.plot(np.abs(box),"k-",label="abs")
    plt.grid()
    plt.title("box")
    plt.legend()

    plt.tight_layout()

    # strdatetime = dt.today().strftime("%Y-%m-%d_%H.%M.%S")
    # np.save("figures/experiments/series3_{}.npy".format(strdatetime),reconstructed_window)
    # print("saved window")
    # plt.savefig("figures/experiments/series3_{}.png".format(strdatetime))
    # print("saved figure")
    plt.show()