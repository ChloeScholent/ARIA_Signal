import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

audio_file_1 = 'XC702143_left.wav'
audio_file_2 = 'XC972996_left.wav'

# Read audio files
sample_rate_1, audio_data_1 = wavfile.read(audio_file_1)
sample_rate_2, audio_data_2 = wavfile.read(audio_file_2)

Fs = sample_rate_1  # use one sample rate
Fs_2 = sample_rate_2

x = audio_data_1
x_2 = audio_data_2

# First investigations
N = np.size(x)
N_2 = np.size(x_2)
# print(f"Sampling Frequency : {Fs} Hz")
# print(f"Number of samples : {N}")
# print(f'Length of the signal : {N/Fs}s')


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

# We can zoom on the signal bandwidth : here between -5Hz and +5Hz
X,f=my_fft(x,Fs)
plt.figure("Zoom between -5Hz and +5Hz")
plt.plot(f,np.abs(X)**2)
plt.xlim((0,10000))
plt.xlabel('Frequency (Hz)')
plt.title('$|X[k]|^2$')
plt.show()




def band_pass_filter(x, fc, Fs):
    # fc is a tuple (low_cut, high_cut)
    wc = [f / (Fs/2) for f in fc]  # normalize each frequency
    b, a = signal.butter(4, wc, btype='bandpass')
    y = signal.filtfilt(b, a, x)
    return y


# Band-pass filter with cut frequencies fc1 = 1Hz and fc2 = 3Hz
fc=(2700,4500)
y=band_pass_filter(x,fc,Fs)

# Plot in the time domain
plt.figure("Band-pass filter - Time domain")
plt.plot(t,x)
plt.plot(t,y)
plt.xlim((0,(N-1)/Fs))
plt.xlabel('Time (seconds)')
plt.legend(('Original signal', 'Filtered signal'))
plt.show()

# Plot in the frequency domain
X,f=my_fft(x,Fs)
Y,f=my_fft(y,Fs)
plt.figure("Band-pass filter - Frequency domain")
plt.plot(f,np.abs(X)**2)
plt.plot(f,np.abs(Y)**2)
plt.xlim((0,10000))
plt.xlabel('Frequency (Hz)')
plt.legend(('Original signal', 'Filtered signal'))
plt.show()
    



def my_spectrogram(x,Nw,No,Fs):
    f, t, Sxx = signal.stft(x, fs=Fs,nperseg=Nw, noverlap=No, nfft=4*Nw)
    return f,t,Sxx

Nw=64 # Window length
No=60 # Overlap length
f, t, Sxx = my_spectrogram(x,Nw,No,Fs)
plt.figure("Spectrogram")
plt.pcolormesh(t, f, np.abs(Sxx)**2,shading='auto')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()    
