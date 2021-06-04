# this module is to be imported
import numpy as np
from scipy.fft import fftshift,fft

NTAP = 4
NCHAN = 1025 # there's a plus one here!!!
LBLOCK = 2*(NCHAN-1)
SINC = np.sinc(np.arange(-NTAP/2,NTAP/2,1/LBLOCK))
if len(SINC) != NTAP*LBLOCK: raise Exception("incompatible length NTAP, LBLOCK")
SINC_HANNING = SINC * np.hanning(NTAP*LBLOCK)
BOXCAR_0 = fftshift(fft(fftshift(SINC)))
PI = np.pi

# useful for quantization loss optimizations
SINPI = 0
SIN2PI = 0
SIN3PI = 0
COSPI = -1
COS2PI = 1
COS3PI = -1
