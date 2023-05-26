# pfb-mod
Modifying the Polyphase Filter Bank to make it robust to quantization effects

The polyphase filter bank (PFB) is a widely used digital signal processing tool used for channelizing input from radio telescopes. Quantization of the channelized signal leads to blow ups in error. We present a practical method for inverting the PFB with minimal quantization-induced error that requires as little as 3\% extra bandwidth.

## Outline of code
- `pfb.py` contains fuctions to perform the forward and inverse PFB, and methods to quantize the inverse.
- `helper.py` utility functions used to analyzing the quantization errors induced in the quantized iPFB. 
- `conjugate_gradient.py` contains functions to optimize the chi-squared value of the iPFB. 
- `matrix_operations` is a helper that wraps PFB related operations in a way that makes them look like the linear operators that they really are. These are used in conjugate gradient descent algorithm. 
- `optimal_wiener_thresh.py` finds the optimal Wiener threshold parameter. 
- `plots/plotall.sh` generates all plots (which can be found in `plots/img`). 


## Dependencies / libraries used in scritps:
- Jax, for autograd custom gradient descent functions

The usual suspects
- Numpy
- Scipy (``scipy.signal``, ``scipy.optimize``)
- Matplotlib

## Optimal quantization

Let $X$ be a gaussian random variable with $\mu=0$ and $\sigma=1$, and let $(x_n)$ be a sequence of i.i.d realizations of $X$. If we would like to quantize this signal to four bits, what is the optimal quantization interval? This is not a mathematically precise question because it depends on what your optimizing. Lets say we wish to minimize the expected magnitude squared of the residual $R=X-\tilde X$, where $\tilde X$ is a quantized signal. 


Let $(y_n)$ be the normalized FFT of $(x_n)$ defined as follows

$$y_n = \frac{1}{\sqrt N} \sum_{k=0}^N \exp(-2\pi i nk)x_k$$


## Optimizing the Inverse PFB using extra information

In `conjugate_gradient.py`, we have code that lets us optimize the inverse PFB based on some added information. 

We perform conjugate gradient descent on a matrix equation of the form

$$B x = u$$

The equation we are minimizing, the chi-squared equation, takes the form

$$\chi^2 = (d - Ax)^T N^{-1} (d - Ax)$$

Taking a derivative wrt to the model ($x$) and setting that to zero we get

$$\frac{d\chi^2}{dx} = -2A^TN^{-1}(d - Ax) = 0 \Rightarrow A^TN^{-1}(d - Ax) = 0$$



## Plots


![eigenvalues_ntap4_lblock2048](https://user-images.githubusercontent.com/21654151/203455742-f0ebf621-e0f9-4e48-9a3a-dccbf97674ef.png)

![four_segments_sinc_hanning](https://user-images.githubusercontent.com/21654151/203455746-38a2cdd6-92e9-438d-876d-f678bebd0301.png)

![RMSE_analytic_lblock](https://user-images.githubusercontent.com/21654151/203455753-d2261038-492c-466b-ae5e-bc61d6599e82.png)

![RMSE_conjugate_gradient_descent_0percent](https://user-images.githubusercontent.com/21654151/203455757-4ae08669-8cd8-4331-b5a1-93c7aed17e1e.png)
![RMSE_conjugate_gradient_descent_1percent](https://user-images.githubusercontent.com/21654151/203455761-929a4faa-30aa-46a7-b1ab-7973143dc5f7.png)
![RMSE_conjugate_gradient_descent_3percent](https://user-images.githubusercontent.com/21654151/203455767-0c14d5c7-9b33-4d44-9378-dd1d76bb6c65.png)
![RMSE_conjugate_gradient_descent_5percent](https://user-images.githubusercontent.com/21654151/203455770-11e6bcf7-00df-4c0a-8746-0c0337bcfd64.png)
![RMSE_log_virgin_IPFB_residuals_wiener](https://user-images.githubusercontent.com/21654151/203455777-7c5887ed-ad4c-44ef-b775-4837420ff431.png)
![rmse_wiener_eigenspec](https://user-images.githubusercontent.com/21654151/203455779-4333e544-23e0-44f4-8efd-46122d236c99.png)
![RMSE_wiener_lblock](https://user-images.githubusercontent.com/21654151/203455781-8267bab8-b29f-4ec7-a033-51fead169d24.png)
![RMSE_wiener_long_time](https://user-images.githubusercontent.com/21654151/203455783-0cbf4100-1c35-4e64-81c7-d43ec8c1025d.png)
![sidelobes](https://user-images.githubusercontent.com/21654151/203455787-b27daaf6-217b-424e-be5b-8bc56508d2ab.png)

<img width="653" alt="sample_prior_algorithm" src="https://user-images.githubusercontent.com/21654151/203611692-27ba0b03-9379-4e8c-9bf7-3ca640202aaa.png">
<img width="646" alt="sw_extract_tikz" src="https://user-images.githubusercontent.com/21654151/203611694-7e219311-2943-434a-a76a-f502305af928.png">
<img width="645" alt="sw_matrix" src="https://user-images.githubusercontent.com/21654151/203611696-601cb03a-4f24-45fa-835f-aa4a97f65d08.png">

Standardise the variable naming convention without breaking code
- nblock = nframe + ntap - 1 
- lblock -> lframe
- nchan is gucci
- k -> nblock
- Perhaps get rid of nblock altogether? It may cause unnecessary confusion. Perhaps nframe should take nblock's definition and nframe's previous def should be abolished (nframe currently is spec.shape[0]).

