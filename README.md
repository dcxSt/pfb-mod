# pfb-mod
Modifying the Polyphase Filter Bank to make it robust to quantization effects

The polyphase filter bank (PFB) is a widely used digital signal processing tool used for channelizing input from radio telescopes. Quantization of the channelized signal leads to blow ups in error. We present a practical method for inverting the PFB with minimal quantization-induced error that requires as little as 3\% extra bandwidth.

[Link to the write-up](https://www.overleaf.com/1895914395bjkqwzjzgkrp) 

### Dependencies / libraries used in scritps:
- Jax, Autograd and gradient descent

The usual suspects
- Numpy
- Scipy (``scipy.signal``, ``scopy.fft``, ``scipy.optimize``)
- Matplotlib


