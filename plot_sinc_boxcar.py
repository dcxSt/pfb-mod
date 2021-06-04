import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft,fftshift
from constants import *


k = 0.0003663 # the boundary of nice sinc is between 0.0003663 and 0.0003662
boxcar = np.concatenate([np.zeros(int(NTAP*LBLOCK*((1-k)/2))),np.ones(int(NTAP*LBLOCK*k)),np.zeros(int(NTAP*LBLOCK*((1-k)/2)))])

sinc = np.sinc(np.linspace(-2,2,2*int(NTAP*LBLOCK/2)))

# plot the windows and their ffts
plt.subplots(figsize=(14,7))
plt.subplot(2,3,1)
plt.plot(boxcar)
plt.title("Boxcar",fontsize=20)

plt.subplot(2,3,2)
plt.plot(boxcar[int(NTAP*LBLOCK/2)-4:int(NTAP*LBLOCK/2)+3])
plt.title("Boxcar Zoom",fontsize=20)

plt.subplot(2,3,3)
plt.plot(fftshift(ifft(fftshift(boxcar))))
plt.title("(shifted) ifft Boxcar",fontsize=20)

plt.subplot(2,3,4)
plt.plot(sinc)
plt.title("Sinc",fontsize=20)

plt.subplot(2,3,5)
plt.plot(fftshift(fft(fftshift(sinc))))
plt.title("(shifted) fft Sinc",fontsize=20)
plt.subplot(2,3,6)
plt.plot(fftshift(fft(fftshift(sinc)))[int(NTAP*LBLOCK/2)-10:int(NTAP*LBLOCK/2)+10])
plt.title("fft Sinc Zoom",fontsize=20)

plt.tight_layout()
plt.show()
