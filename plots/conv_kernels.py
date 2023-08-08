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



l=len(np.fft.irfft(mat_eig[0,:])) # length of irfft'd mat_eig row

col=1024
fig,ax=plt.subplots(2,3,figsize=(10,4))
ax[0,0].set_title(f"Time domain kernel\nColumn #{col} of 2048")
ax[0,0].plot(np.fft.irfft(1/mat_eig[col,:]),".-",color=c0,markersize=1.0,linewidth=0.5)
ax[1,0].set_title(f"Expected error^2 multiplier {sum(np.fft.irfft(1/mat_eig[col,:])**2):.1f}")
ax[1,0].fill_between(np.arange(l),np.fft.irfft(1/mat_eig[col,:])**2,alpha=.5,color=c1)


col=1000
ax[0,1].set_title(f"Time domain kernel\nColumn #{col} of 2048")
ax[0,1].plot(np.fft.irfft(1/mat_eig[col,:]),".-",color=c0,markersize=1.0,linewidth=0.5)
ax[1,1].set_title(f"Expected error^2 multiplier {sum(np.fft.irfft(1/mat_eig[col,:])**2):.1f}")
ax[1,1].fill_between(np.arange(l),np.fft.irfft(1/mat_eig[col,:])**2,alpha=.5,color=c1)



col=10
ax[0,2].set_title(f"Time domain kernel\nColumn #{col} of 2048")
ax[0,2].plot(np.fft.irfft(1/mat_eig[col,:]),".-",color=c0,markersize=1.0,linewidth=0.5)
ax[1,2].set_title(f"Expected error^2 multiplier {sum(np.fft.irfft(1/mat_eig[col,:])**2):.1f}")
ax[1,2].fill_between(np.arange(l),np.fft.irfft(1/mat_eig[col,:])**2,alpha=.5,color=c1)

fig.suptitle("Unfiltered Kernels",fontsize=20)

# remove xticks
ax[0,0].set_xticks([])
ax[0,1].set_xticks([])
ax[0,2].set_xticks([])
ax[1,0].set_xticks([])
ax[1,1].set_xticks([])
ax[1,2].set_xticks([])


fig.tight_layout()
plt.savefig("img/kernels_unfiltered.png",dpi=400)
plt.show(block=True)

### WIENER


phi=0.07
def wien(Fw:np.ndarray,phi:float):
    return abs(Fw)**2 * (1+phi)**2 / (abs(Fw)**2 + phi**2)

colors=plt.get_cmap('Set2').colors # get list of RGB color values
c0=colors[0]
c1=colors[1]

col=1024
fig,ax=plt.subplots(2,3,figsize=(10,4))
ax[0,0].set_title(f"Time domain kernel\nColumn #{col} of 2048")
wienarr = wien(mat_eig[col,:],phi)
ax[0,0].plot(np.fft.irfft(wienarr/mat_eig[col,:]),".-",color=c0,markersize=1.0,linewidth=0.5)
ax[1,0].set_title(f"Expected error^2 multiplier {sum(np.fft.irfft(wienarr/mat_eig[col,:])**2):.1f}")
ax[1,0].fill_between(np.arange(l),np.fft.irfft(wienarr/mat_eig[col,:])**2,alpha=.5,color=c1)


col=1000
ax[0,1].set_title(f"Time domain kernel\nColumn #{col} of 2048")
wienarr = wien(mat_eig[col,:],phi)
ax[0,1].plot(np.fft.irfft(wienarr/mat_eig[col,:]),".-",color=c0,markersize=1.0,linewidth=0.5)
ax[1,1].set_title(f"Expected error^2 multiplier {sum(np.fft.irfft(wienarr/mat_eig[col,:])**2):.1f}")
ax[1,1].fill_between(np.arange(l),np.fft.irfft(wienarr/mat_eig[col,:])**2,alpha=.5,color=c1)



col=10
ax[0,2].set_title(f"Time domain kernel\nColumn #{col} of 2048")
wienarr = wien(mat_eig[col,:],phi)
ax[0,2].plot(np.fft.irfft(wienarr/mat_eig[col,:]),".-",color=c0,markersize=1.0,linewidth=0.5)
ax[1,2].set_title(f"Expected error^2 multiplier {sum(np.fft.irfft(wienarr/mat_eig[col,:])**2):.1f}")
ax[1,2].fill_between(np.arange(l),np.fft.irfft(wienarr/mat_eig[col,:])**2,alpha=.5,color=c1)
fig.suptitle("Wiener Filtered Kernels",fontsize=20)

# remove xticks
ax[0,0].set_xticks([])
ax[0,1].set_xticks([])
ax[0,2].set_xticks([])
ax[1,0].set_xticks([])
ax[1,1].set_xticks([])
ax[1,2].set_xticks([])

fig.tight_layout()
plt.savefig("img/kernels_wiener_filtered.png",dpi=400)
plt.show(block=True)










