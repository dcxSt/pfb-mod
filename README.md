# pfb-mod
Modifying the Polyphase Filter Bank to make it robust to quantization effects

The polyphase filter bank (PFB) is a widely used digital signal processing tool used for channelizing input from radio telescopes. Quantization of the channelized signal leads to blow ups in error. We present a practical method for inverting the PFB with minimal quantization-induced error that requires as little as 3\% extra bandwidth.

[Link to the long write-up](https://www.overleaf.com/1895914395bjkqwzjzgkrp) 
[Link to the article]() 

## Questions
- What is the Weiner filtering doing? Is it just the same thing as setting gausian noise as the prior and then minimizing a chi-squared based on that prior? If so, when the data is completely garbled it doesn't actually help us much... Investigate

## Dependencies / libraries used in scritps:
- Jax, for autograd custom gradient descent functions

The usual suspects
- Numpy
- Scipy (``scipy.signal``, ``scopy.fft``, ``scipy.optimize``)
- Matplotlib

## Todo
- [x] For each vital plot, find + manicure the script that generates it and link it in the caption
- [ ] For other plots, find + manicure the script that generates it and link it in the caption
- [ ] Write succinct article, skip derivations, present results cleanly and put derivatinos in the appendix. 


Less urgent todo
- [ ] Refactor `helper.py` and `constants` to make them simpler (only on the main branch.) 




