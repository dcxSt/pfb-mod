# this module is to be imported
import jax.numpy as np # differentiable numpy library
from jax.numpy.fft import fftshift,fft,rfft 

NTAP = 4
NCHAN = 1025 # there's a plus one here!!!
LBLOCK = 2*(NCHAN-1)
SINC = np.sinc(np.arange(-NTAP/2,NTAP/2,1/LBLOCK))
if len(SINC) != NTAP*LBLOCK: raise Exception("incompatible length NTAP, LBLOCK")
SINC_HAMMING = SINC * np.hanning(NTAP*LBLOCK)
BOXCAR_0 = fftshift(fft(fftshift(SINC)))
BOXCAR_R_4X = rfft(fftshift(np.concatenate([SINC,np.zeros(int(len(SINC)*4.0))])))
BOXCAR_4X_HEIGHT = max(abs(BOXCAR_R_4X[:20])) # used as normalizing factor

PI = np.pi
E = np.exp(1)

# useful for quantization loss optimizations
SINPI = 0
SIN2PI = 0
SIN3PI = 0
COSPI = -1
COS2PI = 1
COS3PI = -1

MAX_SINC = max(SINC)
MIN_SINC = min(SINC)
