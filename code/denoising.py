
import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import os

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

def band_pass_filter(x, fc, Fs):
    # fc is a tuple (low_cut, high_cut)
    wc = [f / (Fs/2) for f in fc]  # normalize each frequency
    b, a = signal.butter(4, wc, btype='bandpass')
    y = signal.filtfilt(b, a, x)
    return y
    
folder = "dataset/"
new_folder = "filtered_dataset/"

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    name, ext = os.path.splitext(file)
    sample_rate, audio_data = wavfile.read(file_path)
    Fs = sample_rate
    x = audio_data
    N = np.size(x)
    t=np.arange(N)/Fs

    # Display of the squared absolute value of the DFT as a function of frequency
    X,f=my_fft(x,Fs)
    plt.figure("Display of the squared absolute value of the DFT as a function of frequency")
    plt.plot(f,np.abs(X)**2)
    plt.xlim((-Fs/2,Fs/2))
    plt.xlabel('Frequency (Hz)')
    plt.title('$|X[k]|^2$')
    plt.show()

    # Band-pass filter
    fc=(4000,5500)
    y=band_pass_filter(x,fc,Fs)

    plt.figure("Band-pass filter - Time domain")
    plt.plot(t,x)
    plt.plot(t,y)
    plt.xlim((0,(N-1)/Fs))
    plt.xlabel('Time (seconds)')
    plt.legend(('Original signal', 'Filtered signal'))
    plt.show()

    # output_file = f'{new_folder}{name}{ext}'
    # # Normalize to 16-bit range for audio
    # y_norm = np.int16((y / np.max(np.abs(y))) * 32767)
    # wavfile.write(output_file, Fs, y_norm)