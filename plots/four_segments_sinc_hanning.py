print("\nINFO: Running four_segments_sinc_hanning.py\n")

# Local import
import sys
sys.path.append("..")
# Libraries
import numpy as np
import matplotlib.pyplot as plt

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
plt.plot(seg0, color="green", linewidth=0.8, label="segment 0")
plt.plot(np.arange(0,lblock), sinc_hann[0:lblock], "--", color="green", alpha=0.5, linewidth=3)
# Segment 1
plt.plot(seg1, color="orange", linewidth=0.8, label="segment 1")
plt.plot(np.arange(lblock,2*lblock), sinc_hann[lblock:2*lblock], "--", color="orange", alpha=0.5, linewidth=3)
# Segment 2
plt.plot(seg2, color="purple", linewidth=0.8, label="segment 2")
plt.plot(np.arange(2*lblock,3*lblock), sinc_hann[2*lblock:3*lblock], "--", color="purple", alpha=0.5, linewidth=3)
# Segment 3
plt.plot(seg3, color="red", linewidth=0.8, label="segment 3")
plt.plot(np.arange(3*lblock,4*lblock), sinc_hann[3*lblock:], "--", color="red", alpha=0.5, linewidth=3)
# Labels, formatting
plt.xlabel("", fontsize=16)
plt.ylabel("", fontsize=16)
plt.tight_layout()
# Optionally save the figure
plt.savefig("four_segments_sinc_hanning.png")
plt.legend()
plt.show(block=True)





