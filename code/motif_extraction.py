import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.signal import find_peaks
import time


# Computes the matrix profile according to the normalized Euclidean distance
def matrix_profile(x,w):
    N=np.size(x)
    d=np.inf*np.ones((N-w,))
    for i in range(N-w):
        x_=x[i:i+w]
        c=fast_distance_profile_nEUC(x,x_)
        c[np.maximum(0,int(i-w)):np.minimum(N-w,int(i+w))]=np.inf
        d=np.minimum(d,c)
    return d

# Optimized implementation of sliding Euclidean distance with Mueen's algorithm
def fast_distance_profile_nEUC(x, p):
    c = np.cumsum(np.concatenate(([0], x)))
    c2 = np.cumsum(np.concatenate(([0], x)) ** 2)
    N = np.size(x)
    Np = np.size(p)

    std_p = np.std(p)
    if std_p == 0:
        # If the segment is constant, return a vector of infinities
        return np.inf * np.ones(N - Np)

    p_ = (p - np.mean(p)) / std_p
    p__ = np.zeros((N,))
    p__[0:Np] = np.flip(p_)
    r = np.real(np.fft.ifft(np.multiply(np.fft.fft(x), np.fft.fft(p__))))
    vari = np.sqrt(Np * (c2[Np:-1] - c2[:N - Np]) - (c[Np:-1] - c[:N - Np]) ** 2)
    d = np.sqrt(np.maximum(2 * Np * (1 - np.divide(r[Np - 1:N - 1], vari)), 0))
    return d


sample_rate, audio_data = wavfile.read("filtered_dataset/filtered_mono_XC707455.wav")
Fs = sample_rate
x = audio_data
N = np.size(x)
t=np.arange(N)/Fs

# Computation and plot of the matrix profile
L=1
m=matrix_profile(x,L)

plt.figure("Matrix profile")
plt.subplot(2,1,1)
plt.plot(x)
plt.xlim([0,np.size(x)])
plt.ylabel('Input signal')
plt.subplot(2,1,2)
plt.plot(m)
plt.xlim([0,np.size(x)])
plt.ylabel('Matrix Profile')
ind=np.argmin(m)
plt.plot(ind,m[ind],'*r')
plt.legend(("Matrix profile","Minimum matrix profile value"))
plt.show()