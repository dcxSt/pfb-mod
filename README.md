# pfb-mod

*This branch is the branch where I rather unceremoniously dumped all of the code, jupyter notebooks, and experiments as backup...*

Modifying the Polyphase Filter Bank to make it robust to quantization effects

The polyphase filter bank (PFB) is a widely used digital signal processing tool used for channelizing input from radio telescopes. Quantization of the channelized signal leads to blow ups in error. We present a practical method for inverting the PFB with minimal quantization-induced error that requires as little as 3\% extra bandwidth.

[Link to the write-up](https://www.overleaf.com/1895914395bjkqwzjzgkrp) 

Dependencies:
- Jax
- Numpy
- Scipy (``scipy.signal``, ``scopy.fft``, ``scipy.optimize``)
- Matplotlib
