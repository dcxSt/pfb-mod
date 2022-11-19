# pfb-mod
Modifying the Polyphase Filter Bank to make it robust to quantization effects

The polyphase filter bank (PFB) is a widely used digital signal processing tool used for channelizing input from radio telescopes. Quantization of the channelized signal leads to blow ups in error. We present a practical method for inverting the PFB with minimal quantization-induced error that requires as little as 3\% extra bandwidth.

## Outline of code
`pfb.py` contains fuctions to perform the forward and inverse PFB, and methods to quantize the inverse.
`helper.py` utility functions used to analyzing the quantization errors induced in the quantized iPFB. 
`conjugate_gradient.py` contains functions to optimize the chi-squared value of the iPFB. 
`matrix_operations` is a helper that wraps PFB related operations in a way that makes them look like the linear operators that they really are. These are used in conjugate gradient descent algorithm. 
`optimal_wiener_thresh.py` finds the optimal Wiener threshold parameter. 
`plots/plotall.sh` generates all the relevant plots. 


## Dependencies / libraries used in scritps:
- Jax, for autograd custom gradient descent functions

The usual suspects
- Numpy
- Scipy (``scipy.signal``, ``scopy.fft``, ``scipy.optimize``)
- Matplotlib




