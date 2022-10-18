import sys
sys.path.append("..")
import helper as h
import numpy as np
from numpy.fft import rfft,fftshift,fftshift,fft
import matplotlib.pyplot as plt

# Choose a photocopy & color-blind friendly colormap
colors=plt.get_cmap('Set2').colors # get list of RGB color values

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
plt.figure(figsize=(12,8))
plt.semilogy(np.abs(short_box_sinc),"-",label="sinc window",color=colors[0])
plt.semilogy(np.abs(short_box),"-",label="sinc hanning window",color=colors[1])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.ylabel("Gain",fontsize=23)
plt.xlabel("Relative Frequency",fontsize=23)
plt.title("Sidelobes of PFB\nSinc and Sinc Hanning",fontsize=30)
plt.grid(which="both")
plt.legend()


plt.savefig("img/sidelobes.png")
plt.show(block=True)











