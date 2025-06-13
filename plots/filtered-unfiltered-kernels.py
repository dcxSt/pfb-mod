import sys
sys.path.append("..")
import helper as h
# Libraries
import numpy as np
import matplotlib.pyplot as plt

# `ntap` is the number of taps
# `lblock` is `2*nchan` which is the number of channels before transform
ntap, lblock = 4, 2048

# A sinc array
sinc = np.sinc(np.linspace(-ntap/2, ntap/2, ntap*lblock))
# Generate the matrix of eigenvalues
mat_eig = h.r_window_to_matrix_eig(sinc * np.hanning(ntap*lblock), 
                                    ntap, lblock)


##################################################################################
###                                    xyz                                     ###
##################################################################################

def wien(Fw:np.ndarray,phi:float):
    return abs(Fw)**2 * (1+phi)**2 / (abs(Fw)**2 + phi**2)
phi=0.07

# Set up matplotlib parameters for clean appearance
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, 
    figsize=(8, 5),
    sharex='col',  # Share x-axis vertically (by column)
    sharey='row'   # Share y-axis horizontally (by row)
)

# Set transparent background
fig.patch.set_alpha(0.0)

# Define colors for aesthetic appeal
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']




col1,col2,col3,col4 = 1024,1022,1000,800
shift=np.fft.fftshift
irfft=np.fft.irfft
# x=np.linspace(0,1024)
# xlims=(512-100,512+100)
# Plot data on each panel
ax1.plot(shift(irfft(1/mat_eig[col1,:])), color=colors[0], linewidth=2, alpha=0.5)
ax2.plot(shift(irfft(1/mat_eig[col2,:])), color=colors[1], linewidth=2, alpha=0.5)
ax3.plot(shift(irfft(1/mat_eig[col3,:])), color=colors[2], linewidth=2, alpha=0.5)
ax4.plot(shift(irfft(1/mat_eig[col4,:])), color=colors[3], linewidth=2, alpha=0.5)

ax1.plot(shift(irfft(wien(mat_eig[col1,:],phi)/mat_eig[col1,:])), color=colors[0], linewidth=1, alpha=1)
ax2.plot(shift(irfft(wien(mat_eig[col2,:],phi)/mat_eig[col2,:])), color=colors[1], linewidth=1, alpha=1)
ax3.plot(shift(irfft(wien(mat_eig[col3,:],phi)/mat_eig[col3,:])), color=colors[2], linewidth=1, alpha=1)
ax4.plot(shift(irfft(wien(mat_eig[col4,:],phi)/mat_eig[col4,:])), color=colors[3], linewidth=1, alpha=1)

# Add panel labels
ax1.set_title(f'Frame index {col1} of 2048', fontsize=12)
ax2.set_title(f'Frame index {col2} of 2048', fontsize=12)
ax3.set_title(f'Frame index {col3} of 2048', fontsize=12)
ax4.set_title(f'Frame index {col4} of 2048', fontsize=12)




# Remove tick labels where specified
# Keep only bottom x-ticks
for ax in [ax1, ax2]:
    ax.tick_params(labelbottom=False)

# Keep only left y-ticks
for ax in [ax2, ax4]:
    ax.tick_params(labelleft=False)

# Add axis labels only where ticks are shown
ax3.set_xlabel('Sample #', fontsize=11)
ax4.set_xlabel('Sample #', fontsize=11)
ax1.set_ylabel('Amplitude (unitless)', fontsize=11)
ax3.set_ylabel('Amplitude (unitless)', fontsize=11)

# Customize grid appearance
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, alpha=0.4, linestyle='--', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.patch.set_alpha(0.0)  # Make subplot backgrounds transparent

# Add suptitle
fig.suptitle('Filtered vs Unfiltered Deconvolution Kernels', 
             fontsize=18, y=0.92)

# Apply tight layout with some padding
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.savefig("./img/filtered-unfiltered-kernels.png")
plt.show()
