# depricated see jupyter notebook : stats how does error carry forward

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

pi = np.pi
exp = np.exp
sqrt = np.sqrt

def get_cum_dist_from_unif(n_runs=1000,n_data=10000,delta=1.0):
    """a random variable x samples uniformly from (-delta/2,delta/2)  
    What is the distribution of the random variable X = sum(x) over n such instances? I couldn't figure it out so this function is to do some experiments."""
    big_x = np.random.uniform(low=-delta/2,high=-delta/2, size=(n_runs,n_data))
    big_x.sum(axis=0)/sqrt(n_runs)
    return big_x

def plot_hist_from_dist(dist,n_runs,b=50):
    n,bins,_ = plt.hist(dist,b)
    plt.show(block=True)
    bin_means = bins[1:] / 2 + bins[:-1] / 2
    gauss = lambda x,sigma,amp:a*exp(-((x)/sigma)**2/2) / sqrt(2*pi*sigma**2) # assumes mean is zero, but not necessarily normalized
    popt,_ = curve_fit(gauss, bin_means, n/n_runs, p0=[1.,1.])
    sigma,amp = popt
    print("sigma = {} \namp {}".format(sigma,amp))
    # plt.show(block=True)
    x = np.linspace(bin_means[0],bin_means[-1],300)
    plt.plot(x, n_runs * gauss(x))
    plt.show(block=True)
    return 

n_runs,n_data = 10000,10000
dist = get_cum_dist_from_unif(n_runs,n_data,delta=1.0)
print("got the dist")
plot_hist_from_dist(dist,n_runs,b=25)
