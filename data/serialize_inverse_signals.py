print("\nINFO: Running rmse_conj_grad_with_quantization_radio_data.py\n")

import sys
sys.path.append("..")
import conjugate_gradient as cg
import numpy as np
import matplotlib.pyplot as plt

   
# Load Radio Data
radio_data_path="/Users/steve/Documents/code/pfb-mod/data/gqrx_20230412_225548_101000000_2500000_fc.raw" # path to radio data
signal = np.fromfile(radio_data_path, dtype="float32")
# normalize it
signal = (signal - np.mean(signal)) / np.std(signal)

# trunkate signal (temporary)
#signal = signal[:3*80*2048]

cg.reconstruct_long_signal(signal,delta=0.3)




