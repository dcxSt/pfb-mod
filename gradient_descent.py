#!/usr/bin/python3.8

import numpy as np
from scipy.fft import fft,rfft,fftshift,irfft,ifftshift
from scipy.signal import gaussian
from eigenspectrum_plotting_library import zero_padding,chop_win # helpers
from constants import *
import helper 

#%% Constants specific to this file
ALPHA = 40.0 # LEARNING RATE
EPSILON = 15.0 # finite difference derivative step
EPOCHS = 500 # Number of epochs
ITERATIONS = 3 # number of steps per epoch
N_DIM = 6
# N_SAMPLES_DIM = 2

# Hyper params for loss function
N_SAMPLES = 256*4
N_THETAS = 260
THRESH = 0.001

#%% more efficient chop win that downsamples first

def chop_win_downsample(w,n_samples=256,ntap=NTAP,lblock=LBLOCK):
    # assumes ntap*lblock == len(w)
    s = np.array(np.linspace(0,lblock*(1-1/n_samples),n_samples),dtype='int') # the sample indices
    return np.reshape(w,(ntap,lblock)).T[s]

#%% loss functions
def quantization_loss_1(window):
    """Loss function for quantization errors
    
    Params : window function (e.g. SINC)
    Returns : loss

    the loss is the sum of recipricals of the eigenvalues
    """
    w2d = chop_win(window,NTAP,LBLOCK)
    w2d_padded = zero_padding(w2d)
    ft = np.apply_along_axis(rfft,1,w2d_padded)
    ft_abs_flat = np.abs(ft).flatten()+0.07
    # add cnst term (0.07) so that nothing explodes in sum below
    loss = np.sum(1/ft_abs_flat)
    return loss # this brings current value to zero # quite bad

def q_loss_method_0(vals,thresh=THRESH):
    return np.count_nonzero(vals<=thresh)

def q_loss_method_1(vals):
    """Loss function for quantization errors

    Params : 
        vals : float[] -- array of samples of frequency eigenvalues (norm squared) 
    Returns : 
        loss : float -- the loss function
    
    The loss is caluclated below -- ni is the number of values below 0.i
    """
    vsqrt = np.sqrt(vals)
    n1 = np.count_nonzero(vsqrt<=0.1)
    n2 = np.count_nonzero(vsqrt<=0.2) - n1
    n3 = np.count_nonzero(vsqrt<=0.3) - n2 - n1
    n4 = np.count_nonzero(vsqrt<=0.4) - n1 - n2 - n3
    n5 = np.count_nonzero(vsqrt<=0.5) - n1 - n2 - n3 - n4 
    return n1 + 0.6*n2 + 0.3*n3 + 0.15*n4 + 0.1*n5

def q_loss_method_2(vals):
    """Loss function for quantization errors

    Params : 
        vals : float[] -- array of samples of frequency eigenvalues (norm squared) 
    Returns : 
        loss : float -- the loss function
    
    The loss is caluclated below -- ni is the number of values below 0.i
    """
    n1 = np.count_nonzero(vals<=0.1)
    n2 = np.count_nonzero(vals<=0.2) - n1
    n3 = np.count_nonzero(vals<=0.3) - n2 - n1
    n4 = np.count_nonzero(vals<=0.4) - n1 - n2 - n3
    n5 = np.count_nonzero(vals<=0.5) - n1 - n2 - n3 - n4 
    return n1 + 0.6*n2 + 0.3*n3 + 0.15*n4 + 0.1*n5

def q_loss_method_3(vals):
    return np.sum(1/(0.1+np.sqrt(vals)))

def q_loss_method_4(vals):
    return np.sum(1/(0.1+vals))


def quantization_sample_single_fourier_value_at_pi(window,n_samples=128):
    """Loss function for quantizaion errors
    
    Params : float[] window function (e.g. SINC)
    Resturns : loss

    The loss is evaluated as follows
        chunk up the window
        samples k chunks (where k is a number like 256 for instance)
        evaluate y=|W(x)| at x=PI -- where W is the RDFT of [a,b,c,d,0,0,0,...,0] normalized so that x is between 0 and PI
        return 1/(y+0.1) -- so 10 if it's 0, 5 if it's 0.1, 3. if it's 0.2, 2.5 if it's 0.3 
        also try return 1/(y**2+0.1)
    """
    p2d = chop_win_downsample(window,n_samples=n_samples)
    min_guess_ft = lambda arr: sum(arr * np.array((1,-1,1,-1)))# takes as input an array with four elements
    vals = np.apply_along_axis(min_guess_ft,1,p2d)**2 # square or else negative values...
    return vals

def quantization_sample_fourier_values(window,n_samples=128,n_thetas=50,thetas_cust=None):
    """Loss function for quantization errors
    
    Params, Returns same as above
    
    The loss is evaluated as follows
        Same as above except, evaluates multiple points on array
    """
    # n_samples = 128 # the number of columns to sample in the 
    # n_thetas = 50 # number of thetas to sample from 
    
    sin = np.sin
    cos = np.cos

    p2d = chop_win_downsample(window,n_samples=n_samples)
    symmetrise = lambda m: m + m.T + np.diag((1,1,1,1)) # fills diagonal with ones
    def matrix(t):
        m = np.array(((0,cos(t),cos(2*t),cos(3*t)),
                        (0,0,cos(t)*cos(2*t)+sin(t)*sin(2*t),cos(t)*cos(3*t)+sin(t)*sin(3*t)),
                        (0,0,0,cos(2*t)*cos(3*t)+sin(3*t)),
                        (0,0,0,0)))
        return symmetrise(m)

    if type(thetas_cust)!=type(None):theta_arr = thetas_cust 
    else:theta_arr= np.linspace(0,PI,n_thetas)
    p3d = np.repeat(np.array([p2d]),len(theta_arr),axis=0)
    eval_at_t = lambda arr1d,t: np.dot(arr1d,np.dot(matrix(t),arr1d)) # optionally put a square root here

    def eval_at_t_block_2d(idx,p2d,t):
        vals[idx] = np.apply_along_axis(eval_at_t,1,p2d,t)

    vals = np.zeros((len(theta_arr),n_samples))

    for idx,(arr2d,t) in enumerate(zip(p3d,theta_arr)):
        eval_at_t_block_2d(idx,arr2d,t)

    vals = vals.flatten()**2

    return vals

def leak_loss(window):
    leak = BOXCAR_0 - fftshift(fft(fftshift(window)))
    loss = np.linalg.norm(leak)
    return loss

#%% A Gradient descent epoch

def loss_function(window):
    return q_loss_method_0(quantization_sample_fourier_values(window,n_samples=N_SAMPLES,n_thetas=N_THETAS))

# depricated
def nabla_loss_old(window):
    short_box = rfft(window)
    len_short_box = len(short_box)
    window_loss = loss_function(window) 
    
    rng = np.random.default_rng() # no repetes random numbers
    sample_dims = rng.choice(len_short_box,size=N_SAMPLES_DIM, replace=False)
    unitvecs = [np.zeros(len_short_box) for i in range(N_SAMPLES_DIM)]
    
    for vec,i in zip(unitvecs,sample_dims):
        vec[i] += 1

    # take the partial derivative along each unit vector in the sample

    epsilon = EPSILON # distance in finite distance computation
    neighbours_box = np.array([short_box + epsilon*vec for vec in unitvecs]) # finite difference derivative

    # compute the parial derivatives along each unitvec
    # this is the step that takes a long time and needs optimization
    neighbours_window = np.apply_along_axis(irfft,1,neighbours_box) # this also takes time..., ugh
    partial_ds = (np.apply_along_axis(loss_function,1,neighbours_window) - window_loss)/epsilon

    nabla = np.zeros(NTAP*LBLOCK)

    nabla_box = np.zeros(len_short_box)
    for j,partial in zip(sample_dims,partial_ds):
        nabla_box[j] += partial
    nabla = irfft(nabla_box)

    return nabla

# compute the gradient 
def nabla_loss(window):
    box = rfft(fftshift(window))
    short_box = box[:N_DIM] # select the part of the box that matters
    remainder_box = box[N_DIM:] # for now it's 4 but we can make bigger
    len_short_box = len(short_box) # this is the dimension of our space
    window_loss = loss_function(window) 

    unitvecs = np.identity(len_short_box) # the unit vectors
    epsilon = 20.0 # distance in finite distance computation
    
    neighbours_box = np.repeat([short_box],len_short_box,axis=0) + epsilon*unitvecs 
    neighbours_box = np.concatenate((neighbours_box,np.repeat([remainder_box],len_short_box,axis=0)),axis=1)# pad it, optionally could also pad with zeros

    neighbours_window = ifftshift(irfft(neighbours_box))
    partial_ds = (np.apply_along_axis(loss_function,1,neighbours_window) - window_loss)/epsilon
    partial_ds = gaussian(2*len(partial_ds),2.0)[len(partial_ds):] # multiply by gaussian window to supress leaking

    nabla = np.concatenate((partial_ds,np.zeros(len(remainder_box)))) # this one has to be np.zeros
    nabla = ifftshift(irfft(nabla)) # ft nabla to put it into window space

    return np.real(nabla)

def normalize_window(window):
    normalized = window *(MAX_SINC - MIN_SINC) / (max(window)-min(window))
    return normalized

def update_step(window):
    nabla = nabla_loss(window)
    return normalize_window(window - ALPHA*nabla)

def gradient_descent(window,iterations=ITERATIONS):
    w = window.copy()
    loss = loss_function(w)
    print(loss) #trace
    losstrace = [loss]
    for i in range(iterations):
        w = update_step(w) 
        loss = loss_function(w)
        print("loss :",loss)
        losstrace.append(loss)

    return w,np.array(losstrace)

#%% main if run
if __name__ == "__main__":
    # image_eigenvalues(sinc_window()) 
    # Gradient descent
    # curr_window = SINC
    # curr_box = BOXCAR_0
    import os
    wind_name = os.listdir("descent-output")
    wind_name.sort()
    wind_name = wind_name[-1]
    curr_window = np.load("descent-output/"+wind_name) 
    curr_box = helper.window_to_box(curr_window) 

    # curr_window = np.load("./descent-output/window_2021-06-03_18.57.38.npy")
    # curr_box = helper.window_to_box(curr_window)
    windows = [curr_window]
    boxcars = [curr_box] 


    losstrace_array = np.array([])
    import datetime
    for i in range(EPOCHS):
        print("Starting Epoch {}".format(i))
        curr_window,losstrace = gradient_descent(curr_window, iterations=ITERATIONS)
        curr_box = helper.window_to_box(curr_window)
        losstrace_array = np.concatenate([losstrace_array,losstrace])
        # windows.append(curr_window)
        # boxcars.append(curr_box) 
        now = datetime.datetime.today()
        string_now = now.strftime("%Y-%m-%d_%H.%M.%S")
        np.save("./descent-output/window_{}.npy".format(string_now),curr_window) # save the figure

    # display figures
    import matplotlib.pyplot as plt
    from eigenspectrum_plotting_library import image_eigenvalues
    image_eigenvalues(curr_window)
    plt.figure(figsize=(8,8))
    plt.plot(losstrace_array)
    plt.savefig("losstrace.png")
    plt.show()

