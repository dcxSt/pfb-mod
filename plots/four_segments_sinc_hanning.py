print("\nINFO: Running four_segments_sinc_hanning.py\n")

# Local import
import sys
sys.path.append("..")
# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Choose a photocopy & color-blind friendly colormap
colors=plt.get_cmap('Set2').colors # get list of RGB color values

# `ntap` is the number of taps
# `lblock` is `2*nchan` which is the number of channels before transform
ntap, lblock = 4, 2048

# A sinc array
sinc = np.sinc(np.linspace(-ntap/2, ntap/2, ntap*lblock))
sinc_hann = sinc * np.hanning(ntap*lblock)
# A longer sinc array, to display the chunks
sinc_long = np.sinc(np.linspace(-ntap/2, ntap/2, ntap*ntap*lblock))
sinc_hann_long = sinc_long * np.hanning(ntap*ntap*lblock)
# The four quarters (segments) to plot 
seg0 = sinc_hann_long[:ntap*lblock]
seg1 = sinc_hann_long[ntap*lblock:2*ntap*lblock]
seg2 = sinc_hann_long[2*ntap*lblock:3*ntap*lblock]
seg3 = sinc_hann_long[3*ntap*lblock:]

# Plot the figure
plt.figure(figsize=(8.5,7))
plt.title("".format(ntap,lblock),fontsize=22)
# Plot the sinc hanning window segments
# Segment 0
#plt.plot(seg0, color="green", linewidth=0.8, label="segment 0")
x,y,c = np.arange(0,lblock),sinc_hann[0:lblock],colors[0]
plt.plot(x,y,"--", color=c, linewidth=3)
plt.fill_between(x,y,step="pre",alpha=0.4,color=c)
# Segment 1
#plt.plot(seg1, color="orange", linewidth=0.8, label="segment 1")
x,y,c = np.arange(lblock,2*lblock),sinc_hann[lblock:2*lblock],colors[1]
plt.plot(x,y, "--", color=c,linewidth=3)
plt.fill_between(x,y,step="pre",alpha=0.4,color=c)
# Segment 2
x,y,c = np.arange(2*lblock,3*lblock),sinc_hann[2*lblock:3*lblock],colors[2]
#plt.plot(seg2, color="purple", linewidth=0.8, label="segment 2")
plt.plot(x,y,"--",color=c,linewidth=3)
plt.fill_between(x,y,step="pre",alpha=0.4,color=c)
# Segment 3
#plt.plot(seg3, color="red", linewidth=0.8, label="segment 3")
x,y,c = np.arange(3*lblock,4*lblock),sinc_hann[3*lblock:4*lblock],colors[3]
plt.plot(x,y,"--",color=c,linewidth=3)
plt.fill_between(x,y,step="pre",alpha=0.4,color=c)
# Plot four big dots
dt = 331 # How far along to plot the dots relative to each resp. segment
times=[2048*i+dt for i in range(4)]
plt.plot(times,sinc_hann[times],'ko',ms=8)

# Labels, formatting
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.xlabel("Channel #", fontsize=16)
plt.ylabel("Sinc Hanning value", fontsize=16)
plt.title("Four segments of Sinc Hanning",fontsize=22)
plt.tight_layout()
# Optionally save the figure
plt.savefig("img/four_segments_sinc_hanning.png",dpi=400)
plt.legend()
plt.show(block=True)





