"""
Created on 2021.06.03 16:08:57
Author : Stephen Fay

Some helper functions
"""

# %env JAX_ENABLE_x64=1 # not sure what this does yet, but it might be important
import jax.numpy as jnp # differentiable numpy library
from jax import numpy as jnp 
from jax.numpy.fft import fft,ifft,fftshift,ifftshift,rfft,irfft
from jax import custom_jvp
from constants import SINC,NTAP,LBLOCK,BOXCAR_4X_HEIGHT
import numpy # numpy is required for cheb_win 


#%% custom differentiable JAX functions, and their derivatives

@custom_jvp
def exp(x):
    return jnp.exp(x)

exp.defjvps(lambda x_dot, primal_out, x: x_dot * jnp.exp(x))

@custom_jvp
def ln(x): # assumes x > 0
    return jnp.log(x)

ln.defjvps(lambda x_dot, primal_out, x: x_dot / x)

@custom_jvp
def ln_safe(x): # assumes x >= 0
    return jnp.log(x+10**(-20))

ln_safe.defjvps(lambda x_dot, primal_out, x: x_dot / (x+10**(-20)))

@custom_jvp
def log10(x): # assumes x > 0
    return jnp.log10(x)

log10.defjvps(lambda x_dot, primal_out, x: x_dot / (x*jnp.log(10)))


@custom_jvp
def log10_safe(x): # assumes x >= 0, useful when you take the log of array with zeros
    return jnp.log10(x+10**(-20)) # can go up to -37 in theory

log10_safe.defjvps(lambda x_dot, primal_out, x: x_dot / ((x+10**(-20))*jnp.log(10)))

@custom_jvp
def log10_safe_2(x): # assumes x >= 0
    return jnp.log10(x+10**(-10))

log10_safe_2.defjvps(lambda x_dot, primal_out, x: x_dot / ((x + 10**(-10))*jnp.log(10)))


#%% fourier transforms and their inverses
def window_to_box(window):
    return fftshift(fft(fftshift(window))) # gets taller

def box_to_window(box):
    return ifftshift(ifft(ifftshift(box))) # gets smaller

def window_pad_to_box(window,pad_factor=4.0):
    # pad the window then fft
    padded_window = jnp.concatenate([window,jnp.zeros(int(len(window)*pad_factor))])
    return window_to_box(padded_window) 

def box_to_window_pad(large_box,len_win):
    return box_to_window(large_box)[:len_win]

def window_pad_to_box_rfft(window,pad_factor=4.0):
    padded_window = jnp.concatenate([window,jnp.zeros(int(len(window)*pad_factor))])
    return rfft(fftshift(padded_window))

def box_to_window_pad_rfft(large_box,len_win):
    return ifftshift(irfft(large_box))[:len_win]

#%% involved in gradient descent
# get a spline function for x,y for the log sidelobes of a window with pad 4.0 x len(window)
def get_spline_func(window):
    half_box = window_pad_to_box_rfft(window,4.0)
    half_box = abs(half_box)
    width = 13 # the width of the boxcar (yes this is hardcoded, can find with peakfinder too)
    log_lobes = jnp.log10(half_box[width:])
    x = [] 
    y = [] 
    count = 0
    ll = log_lobes.copy()
    while len(ll)>40:
        y.append(max(ll[:20]))
        x.append(jnp.argmax(ll[:20]) + 20*count)
        ll = ll[20:]
        count += 1
    x = [0] + x + [len(half_box)-1] # add bits at the end and at the begginning to cover full range of sidelobes
    y = [y[0]] + y + [y[-1]] 
    x,y = jnp.array(x),jnp.array(y)
    from scipy.interpolate import interp1d
    f = interp1d(x, y, kind='cubic')
    return f


# get spline x and y arrays for the log sidelobes of a window with pad 4.0 x len(window)
def get_spline_arr(window):
    f = get_spline_func(window)
    half_box = abs(window_pad_to_box_rfft(window,4.0)[13:])
    x_new = jnp.arange(len(half_box)) # add 13 to this to plot with rfft boxcar
    y_new = f(x_new)
    return x_new,y_new

# moving average, k is over how many neighbours, so k=1 will be av over 3 neighbours
def mav(signal,k=1):
    s = jnp.r_[jnp.ones(k)*signal[0],signal,jnp.ones(k)*signal[-1]] # pad the signal, extend edge values
    w = jnp.ones(2*k+1)
    w /= w.sum()
    return jnp.convolve(w,s,mode="valid")


#%% Chebyshev window

# about 10x faster than previous method that used lists
def cheb_win(coeffs_tail,len_win):
    """
    param coeffs_tail jnp.ndarray : 1d array of chebyshev coefficients, assuems first chebyshev coefficient is 1.0 (the constant term)
    param len_win int : the length of the SINC window (usually 2048 * 4 I think)
    """
    coeffs = jnp.concatenate([jnp.ones(1),coeffs_tail])
    arr2d = jnp.repeat(jnp.array([jnp.linspace(-1,1,len_win)]),len(coeffs),axis=0)
    l = len(coeffs)
    
    diag = numpy.zeros((l,l))
    numpy.fill_diagonal(diag,numpy.arange(l))
    diag = jnp.array(diag)
    arr2d = jnp.matmul(diag,arr2d)
    arr2d = jnp.cos(arr2d)
    
    return jnp.matmul(coeffs,arr2d) 

# almost same as for cheb win
def get_window_from_cosine_coeffs(coeffs,len_win):
    """
    param coeffs_tail jnp.ndarray : 1d array of cosine (think chebyshev) coefficients
    param len_win int : the length of the SINC window (usually 2048 * 4 I think)
    """
    pi = jnp.pi
    arr2d = jnp.repeat(jnp.array([jnp.linspace(-pi/2,pi/2,len_win)]),len(coeffs),axis=0)
    l = len(coeffs)
    
    diag = numpy.zeros((l,l))
    numpy.fill_diagonal(diag,numpy.arange(l))
    diag = jnp.array(diag)
    arr2d = jnp.matmul(diag,arr2d)
    arr2d = jnp.cos(arr2d)
    
    return jnp.matmul(coeffs,arr2d) 


# def cheb_win_skip_old(coeffs_tail,len_win,n_skip):
#     """
#     :param coeff_tail jnp.ndarray: 1d array of chebyshev coeffs
#     """
#     coeffs = jnp.concatenate([jnp.ones(1),jnp.zeros(n_skip),coeffs_tail])
#     l = len(coeffs)
#     arr2d = jnp.repeat(jnp.array([jnp.linspace(-1,1,len_win)]),l,axis=0)
    
#     diag = numpy.zeros((l,l))
#     numpy.fill_diagonal(diag,numpy.arange(l))
#     diag = jnp.array(diag)
#     arr2d = jnp.matmul(diag,arr2d)
#     arr2d = jnp.cos(arr2d)
    
#     return jnp.matmul(coeffs,arr2d) 

# even more performant
def cheb_win_skip(coeffs_tail,len_win,n_skip):
    l = len(coeffs_tail)
    arr2d = jnp.repeat(jnp.array([jnp.linspace(-1,1,len_win)]),len(coeffs_tail),axis=0)
    diag = numpy.zeros((l,l))
    numpy.fill_diagonal(diag,numpy.arange(n_skip+1,n_skip+l+1)) 
    diag = jnp.array(diag)
    arr2d = jnp.matmul(diag,arr2d)
    arr2d = jnp.cos(arr2d)
    
    return jnp.matmul(coeffs_tail,arr2d) + 1.0 
    
    return

# def cheb_win_skip(coeffs_tail,len_win,n_skip):
#     return cheb_win(jnp.concatenate([jnp.zeros(n_skip),coeffs_tail]),len_win)  


def get_modified_sinc_from_cheb(coeffs_tail, win_type=None, sinc=SINC, n_skip=None):
    """Take the cheb coefficients and return amodified SINC window
    param coeffs_tail jnp.ndarray : the cheb coefficients 
    param len_win jnp.float : the length of the window, usually len(SINC)=NTAP*LBLOCK
    param win_type str : either 'hanning' or 'hamming' (or none), the type of window to apply
    
    returns : instance of 1d jnp.ndarray of length len_win representing modified sinc """

    len_win = len(sinc)
    if win_type == None: y = jnp.ones(len_win)
    elif win_type == "hamming": y = jnp.hamming(len_win)
    elif win_type == "hanning": y = jnp.hanning(len_win)
    else: raise Exception("Parameter win_type={} is invalid, must be in \
        \{None, 'hanning', 'hamming'\}".format(win_type))
    
    if not n_skip: return cheb_win(coeffs_tail,len_win) * sinc * y # return sinc multiplied by a window, multiplied by a cheb window
    return cheb_win_skip(coeffs_tail,len_win,n_skip) * sinc * y
    
#%% Metrics for evaluating windows 

def metric_sidelobe_thicknesses(window):
    # determines how thick the boxcar is in multiple ranges
    scale = BOXCAR_4X_HEIGHT
    normed_box = window_pad_to_box_rfft(window,pad_factor=4.0) / scale
    log_box_abs = log10(abs(normed_box))
    l = len(log_box_abs)
    th2 = 100*(max(jnp.where(log_box_abs > -2)[0]) - 13) / l # 13 is the width of the boxcar
    th3 = 100*(max(jnp.where(log_box_abs > -3)[0]) - 13) / l
    th4 = 100*(max(jnp.where(log_box_abs > -4)[0]) - 13) / l
    th5 = 100* max(jnp.where(log_box_abs > -5)[0]) / l # here boxcar width doesn't really matter anymore
    th6 = 100* max(jnp.where(log_box_abs > -6)[0]) / l
    return th2,th3,th4,th5,th6



#%% basic windows, for all windows see windows.py

def sinc_window(ntap=NTAP,lblock=LBLOCK):
    # i don't like np.arange, it's vulnerable to failur, 
    # should be linspace, but this is how R.S. implemented it, and it'll give a (tiny bit) different array
    return jnp.sinc(jnp.arange(-ntap/2,ntap/2,1/lblock)) 

def sinc_hanning(ntap=NTAP,lblock=LBLOCK):
    # different form sinc_window and sinc_hamming, uses numpy not jax.numpy
    return numpy.hanning(ntap*lblock) * numpy.sinc(numpy.linspace(-ntap/2,ntap/2,lblock*ntap))

def sinc_hamming(ntap=NTAP,lblock=LBLOCK):
    return jnp.hamming(ntap*lblock) * sinc_window(ntap,lblock)

#%% helpers for eigenvalue displays
def chop_win(w,ntap=4,lblock=2048):
    """Chop lblock bits of len ntap of window to get ready for DFT"""
    if ntap*lblock!=len(w):raise Exception("len window incompatible")
    return jnp.reshape(w,(ntap,lblock)).T

def zero_padding(w2d,n_zeros=1024):
    pad = jnp.zeros((len(w2d),n_zeros))
    return jnp.concatenate([w2d,pad],axis=1)


def window_to_matrix_eig(w,ntap=NTAP,lblock=LBLOCK,zero_pad_len=1024):
    w2d = chop_win(w,ntap,lblock)
    w2d_padded = zero_padding(w2d,zero_pad_len)
    ft = jnp.apply_along_axis(fft,1,w2d_padded)
    return ft

def matrix_eig_to_window(ft_w2d,ntap=NTAP):
    w2d_padded = jnp.apply_along_axis(ifft,1,ft_w2d)
    w2d = w2d_padded[:,:ntap]
    return jnp.real(jnp.concatenate(w2d.T))

def matrix_eig_to_window_complex(ft_w2d,ntap=NTAP):
    w2d_padded = jnp.apply_along_axis(ifft,1,ft_w2d)
    w2d = w2d_padded[:,:ntap]
    return jnp.concatenate(w2d.T)

def r_window_to_matrix_eig(w,ntap=NTAP,lblock=LBLOCK,zero_pad_len=1024):
    w2d = chop_win(w,ntap,lblock)
    w2d_padded = zero_padding(w2d,zero_pad_len)
    rft = jnp.apply_along_axis(rfft,1,w2d_padded)
    
    return rft

   
if __name__ == "__main__":
    # check that r_window_to_matrix_eig works
    rft = r_window_to_matrix_eig(SINC)
    import matplotlib.pyplot as plt 
    plt.imshow(jnp.abs(rft.T),aspect="auto")
    plt.colorbar()
    plt.show()








