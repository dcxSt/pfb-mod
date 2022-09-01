# this module is to be imported
import jax.numpy as jnp # differentiable numpy library
# from jax.numpy.fft import fftshift,fft,rfft 
from numpy.fft import fftshift,fft,rfft

NTAP = 4
NCHAN = 1025 # there's a plus one here!!!
LBLOCK = 2*(NCHAN-1)
SINC = jnp.sinc(jnp.arange(-NTAP/2,NTAP/2,1/LBLOCK))
if len(SINC) != NTAP*LBLOCK: raise Exception("incompatible length NTAP, LBLOCK")
SINC_HAMMING = SINC * jnp.hamming(NTAP*LBLOCK)
SINC_HANNING = SINC * jnp.hanning(NTAP*LBLOCK)
BOXCAR_0 = fftshift(fft(fftshift(SINC)))
BOXCAR_R_4X = rfft(fftshift(jnp.concatenate([SINC,jnp.zeros(int(len(SINC)*4.0))])))
BOXCAR_4X_HEIGHT = max(abs(BOXCAR_R_4X[:20])) # used as normalizing factor

PI = jnp.pi
E = jnp.exp(1)

# useful for quantization loss optimizations
SINPI = 0
SIN2PI = 0
SIN3PI = 0
COSPI = -1
COS2PI = 1
COS3PI = -1

MAX_SINC = max(SINC)
MIN_SINC = min(SINC)
