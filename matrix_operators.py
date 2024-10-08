"""
Turns our PFB related operations into a the chi-squared matrix notation
"""
import pfb
import numpy as np
from numpy.fft import rfft,irfft




# TODO refactor PFB matrix operators and conj-grad code so that is just does the forward pfb... 

# To build, I'm calling these C instead of A so that old code can coexist, backwards compatible (for now)
def C(x):
    """Applies forward pfb, flatten"""
    return pfb.forward_pfb(x).flatten()

#def C_inv

# ---- old paradigm
def A(x):
    """Applies PFB, irfft's that, flatten."""
    # Forward PFB the Signal
    b = pfb.forward_pfb(x)
    # Inverse Fourier Transform along axis=1
    b = irfft(b)
    # Apply circulant boundary conditions
    b = np.concatenate([b, b[:3, :]], axis=0)
    return b.flatten()

def A_inv(b_flat,lblock=2048):
    """Inverse of A. Reshape the array, rfft, iPFB.
    lblock : int, length of a block (segment)"""
    # Sanity check
    if len(b_flat)/lblock != len(b_flat)//lblock: 
        raise Exception("Dimensions of input do not match lblock!")
    # Reshape array so that it looks like irfft'd pfb output dims
    b = b_flat.reshape((-1,lblock))[:-3,:]
    # Rfft along axis=1
    b = rfft(b)
    return pfb.inverse_pfb(b)

def A_inv_wiener(b_flat, wiener_thresh=0.25,lblock=2048):
    """Inverse of A with wiener filtering. Reshape the array, rfft, iPFB with wiener filter."""
    # Sanity check
    if len(b_flat)/lblock != len(b_flat)//lblock: 
        raise Exception("Dimensions of input do not match lblock!")
    # Reshape array so that it looks like irfft'd pfb output dims
    b = b_flat.reshape((-1,lblock))[:-3,:]
    # Rfft along axis=1
    b = rfft(b)
    return pfb.inverse_pfb(b, wiener_thresh=wiener_thresh)

def A_quantize(x, delta):
    """Takes signal, pfb's it, quantizes, irfft's that."""
    # Forward PFB the signal
    b = pfb.forward_pfb(x)
    # Quantize the filter bank
    # The sqrt is to account for the next IRFFT step
    # b = pfb.quantize(b, np.sqrt(2*(b.shape[1] - 1)) * delta) 
    #b = pfb.quantize_8_bit(b, np.sqrt(2*(b.shape[1] - 1)) * delta) 
    std = ( np.std(np.real(b)) + np.std(np.imag(b)) )/2 # fixed quantization delta scaling with empirical std instead of faulty derived
    b = pfb.quantize_8_bit(b, delta * std)
    # Inverse Fourier Transform
    b = irfft(b) # Same as apply along axis=1
    # Apply circulant boundary conditions
    b = np.concatenate([b, b[:3, :]], axis=0)
    return b.flatten() 

def R(x,lblock=2048):
    """Re-ordering matrix (involution). Useful for Transposes"""
    lx = len(x)
    assert lx/lblock == lx//lblock, "Len x must divide lblock"
    k = lx // lblock # k is number of frames
    out = np.zeros(lx)
    # For each frame, 
    for i in range(k):
        out[i*lblock:(i+1)*lblock] = x[(k-i-1)*lblock:(k-i)*lblock]
    return out

def AT(x): # the transpose of A
    return R(A(R(x)))


