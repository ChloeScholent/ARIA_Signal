import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.io import wavfile
import os
import time

# Naive implementation of sliding normalized Euclidean distance
def distance_profile_nEUC(x,p):
    N=np.size(x)
    Np=np.size(p)
    d=np.zeros((N-Np,))
    p_=(p-np.mean(p))/np.std(p)
    for i in range(N-Np):
        x_=(x[i:i+Np]-np.mean(x[i:i+Np]))/np.std(x[i:i+Np])
        d[i]=np.linalg.norm(x_-p_)
    return d

# Naive implementation of sliding Euclidean distance
def distance_profile_EUC(x,p):
    N=np.size(x)
    Np=np.size(p)
    d=np.zeros((N-Np,))
    for i in range(N-Np):
        d[i]=np.linalg.norm(x[i:i+Np]-p)
    return d

# Optimized implementation of sliding Euclidean distance with Mueen's algorithm
def fast_distance_profile_nEUC(x,p):
    c=np.cumsum(np.concatenate(([0],x)))
    c2=np.cumsum(np.concatenate(([0],x))**2)
    N=np.size(x)
    Np=np.size(p)
    p_=(p-np.mean(p))/np.std(p)
    p__=np.zeros((N,))
    p__[0:Np]=np.flip(p_)
    r=np.real(np.fft.ifft(np.multiply(np.fft.fft(x),np.fft.fft(p__))))
    vari=np.sqrt(Np * (c2[Np:-1]-c2[:N-Np]) -  (c[Np:-1]-c[:N-Np])**2)
    d=np.sqrt(np.maximum(2*Np*(1-np.divide(r[Np-1:N-1],vari)),0))
    return d

# Optimized implementation of sliding Euclidean distance with Mueen's algorithm
def fast_distance_profile_EUC(x,p):
    c2=np.cumsum(np.concatenate(([0],x))**2)
    N=np.size(x)
    Np=np.size(p)
    cp=np.sum(p**2)
    p_=np.zeros((N,))
    p_[0:Np]=np.flip(p)
    r=np.real(np.fft.ifft(np.multiply(np.fft.fft(x),np.fft.fft(p_))))
    d=np.sqrt(np.maximum(c2[Np:-1]-c2[:N-Np]+cp-2*r[Np-1:N-1],0))
    return d

motif_folder = "motif_csv"
folder = "wav_resampled/"
arg = []

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    distances = []
    for motif in sorted(os.listdir(motif_folder)):
        motif_path = os.path.join(motif_folder, motif)
        p = np.loadtxt(motif_path, delimiter=",")

        # Data loading
        _, x1=wavfile.read(file_path)
        Fs=32000

        # First investigations
        N=np.size(x1)
        Np=np.size(p)
        # print("Number of samples in the signals : {N}".format(**locals()))
        # print("Number of signals in the pattern : {Np}".format(**locals()))



        # Example of pattern detection with DTW distance
        d = fast_distance_profile_EUC(x1, p)  # use a step to reduce computation time
        # plt.figure("Distance profile with Euclidean distance")
        # plt.subplot(2,3,(2,3))
        # plt.plot(x1)
        # plt.ylabel('Input signal 1')
        # plt.xlim([0, N])
        # plt.subplot(2,3,4)
        # plt.plot(p)
        # plt.subplot(2,3,(5,6))
        # plt.plot(d)
        # plt.xlim([0, N])
        # plt.title("Euclidean distance profile")
        distances.append(sum(d))
    arg.append(np.argmin(distances))

print(len(arg))

# x=0
# for motif in sorted(os.listdir(motif_folder)):
#     motif_path = os.path.join(motif_folder, motif)
#     x+=1
#     if x==7:
#         print(motif_path)