# What happens to quantization error when you fourier transform?
# before figuring out analytically we experiment computationally to find out what to expect.
import numpy as np
from numpy.fft import fft,rfft,irfft,ifft

def quantize_uniform_noise(signal,delta):
    # input signal simulated getting 'quantized' by adding noise
    noise = np.random.uniform(-delta/2,delta/2,len(signal))
    return signal + noise

def mse(signal,noisy_signal):
    return np.mean(abs(signal - noisy_signal)**2)

if __name__=="__main__":
    mse_signals = []
    mse_rft_signals = []
    sqrt = np.sqrt
    delta = sqrt(12) # 1.0
    n = 100000
    print("Simulating the effects of quantization on signals and their fourier transforms...\n\n")
    for i in range(1000):
        signal = np.random.normal(0,1,n)
        noisy_signal = quantize_uniform_noise(signal,delta)
        mse_signal = mse(signal,noisy_signal)
        mse_signals.append(mse_signal)

        rft_s = rfft(signal)/sqrt(n)
        rft_ns = rfft(noisy_signal)/sqrt(n)
        mse_rft_signal = mse(rft_s,rft_ns)
        mse_rft_signals.append(mse_rft_signal)
    print("mean squared error for the singal is {}\nmean squared error for the rftsig is {}".format(np.mean(mse_signals),np.mean(mse_rft_signals)))
    print("")
    print("the respective stds are {} and {}".format(np.std(mse_signals),np.std(mse_rft_signals)))

