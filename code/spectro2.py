import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy import signal
import os

def my_spectrogram(x,Nw,No,Fs):
    f, t, Sxx = signal.stft(x, fs=Fs,nperseg=Nw, noverlap=No, nfft=4*Nw)
    return f,t,Sxx

sample_rate, audio_data = wavfile.read("filtered_dataset/filtered_mono_XC707455.wav")
Fs = sample_rate
x = audio_data
N = np.size(x)
t=np.arange(N)/Fs


Nw=4 # Window length
No=3 # Overlap length
f, t, Sxx = my_spectrogram(x,Nw,No,Fs)
plt.figure("Spectrogram")
plt.pcolormesh(t, f, np.abs(Sxx)**2,shading='auto')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()               




