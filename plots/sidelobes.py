"""Images the eigenvalues of a window function

Parameters
----------
ntap : integer
    Number of taps
lblock : integer
    Length of bloc. (lblock = 2*nchan)
w : ndarray 
    Window, size ntap times lblock
"""

import sys
sys.path.append("..")
import helper as h
import numpy as np
from numpy.fft import rfft,fftshift,fftshift,fft
import matplotlib.pyplot as plt

ntap = 4
lblock = 2048
sinc = np.sinc(np.arange(-ntap/2,ntap/2,1/lblock))
sinc_hann = h.sinc_hanning(ntap,lblock)

width=250
box = h.window_pad_to_box(sinc_hann,10.0)
short_box = box[int(len(box)/2-width):int(len(box)/2+width)]
box_sinc = h.window_pad_to_box(sinc,10.0)
short_box_sinc = box_sinc[int(len(box_sinc)/2-width):int(len(box_sinc)/2+width)]
scale = max(np.abs(short_box)) # Now we can scale everyone down to where to peak in logplot is zero
box,short_box,box_sinc,short_box_sinc = box/scale,short_box/scale,box_sinc/scale,short_box_sinc/scale

### plot the sidelobes 
plt.figure(figsize=(6,4))
plt.semilogy(np.abs(short_box_sinc),"b-",alpha=0.7,label="sinc")
plt.semilogy(np.abs(short_box),"k-",alpha=0.7,label="sinc hanning")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.ylabel("Gain")
plt.xlabel("Relative Frequency")
plt.title("Sidelobes of PFB\nSinc and Sinc Hanning")
plt.grid(which="both")
plt.legend()


plt.show(block=True)











