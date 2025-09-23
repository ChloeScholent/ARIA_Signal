import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

sample_rate, audio_data = wavfile.read("Chardonneret_right.wav")

# Data loading
x=audio_data
Fs=sample_rate

# First investigations
N=np.size(x)
print("Sampling Frequency : {Fs} Hz".format(**locals()))
print("Number of samples : {N}".format(**locals()))


# x[n] as a function of the sample n
n=np.arange(N)
plt.figure("x[n] as a function of the sample n")
plt.plot(n,x)
plt.xlim((0,N-1))
plt.xlabel('$n$ (samples)')
plt.title('$x[n]$')
plt.show()



# x[n] as a function of the time t[n]
t=np.arange(N)/Fs
plt.figure("x[n] as a function of the time t[n]")
plt.plot(t,x)
plt.xlim((0,(N-1)/Fs))
plt.xlabel('Time (seconds)')
plt.title('$x[n]$')
plt.show()


# Function to compute the centered Fast Fourier Transform (FFT)
def my_fft(x,Fs):
    N=np.size(x)
    #Computation of the FFT
    X=np.fft.fft(x)
    X=np.fft.fftshift(X)
    # Computation the frequency vector
    f=np.fft.fftfreq(N, d=1/Fs)
    f=np.fft.fftshift(f)
    return X,f

# Display of the squared absolute value of the DFT as a function of frequency
X,f=my_fft(x,Fs)
plt.figure("Display of the squared absolute value of the DFT as a function of frequency")
plt.plot(f,np.abs(X)**2)
plt.xlim((-Fs/2,Fs/2))
plt.xlabel('Frequency (Hz)')
plt.title('$|X[k]|^2$')
plt.show()

