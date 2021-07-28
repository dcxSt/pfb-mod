import sys, os
sys.path.append("..")
import helper as h
from constants import SINC
import numpy as np
import matplotlib.pyplot as plt

eigengrid = h.r_window_to_matrix_eig(SINC) # h.window_pad_to_box_rfft(SINC,pad_factor=10.0)
eigengrid_hann = h.r_window_to_matrix_eig(SINC * np.hanning(len(SINC)))
eigengrid_hamm = h.r_window_to_matrix_eig(SINC * np.hamming(len(SINC)))

plt.imshow(abs(eigengrid),aspect="auto")
plt.show(block=True)

plt.subplots(figsize=(10,5))
plt.subplot(121)
plt.plot(np.mean(1/abs(eigengrid**2),axis=1),".",label="SINC")
plt.plot(np.mean(1/abs(eigengrid_hann**2),axis=1),".",alpha=0.4,label="sinc hanning")
plt.plot(np.mean(1/abs(eigengrid_hamm**2),axis=1),".",alpha=0.4,label="sinc hamming")
plt.grid()
plt.legend()

plt.subplot(122)
plt.plot(np.mean(1/abs(eigengrid[:1000]**2),axis=1),".",label="sinc")
plt.plot(np.mean(1/abs(eigengrid_hann[:1000]**2),axis=1),".",alpha=0.4,label="sinc hanning")
plt.plot(np.mean(1/abs(eigengrid_hamm[:1000]**2),axis=1),".",alpha=0.4,label="sinc hamming")
plt.tight_layout()
plt.show(block=True)


