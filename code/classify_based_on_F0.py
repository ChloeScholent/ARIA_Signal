import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


# Function to compute the centered Fast Fourier Transform (FFT)
def my_fft(x, Fs):
    N = np.size(x)
    X = np.fft.fft(x)
    X = np.fft.fftshift(X)
    f = np.fft.fftfreq(N, d=1/Fs)
    f = np.fft.fftshift(f)
    return X, f

F0 = []
folder = "dynamically_filtered_dataset/"
Fs = 32000  # Sampling frequency (Hz)
motif_dict = {}

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)

    # Load motif
    sample_rate, audio_data = wavfile.read(file_path)
    Fs = sample_rate
    x = audio_data
    N = np.size(x)
    t=np.arange(N)/Fs
    #print(f"\nMotif: {file} | Samples: {len(motif)}")

    # Compute FFT
    X, f = my_fft(x, Fs)

    # Compute power spectrum
    power_spectrum = np.abs(X) ** 2

    # Ignore DC (0 Hz) and find the frequency with maximum power
    mid = len(f) // 2
    power_spectrum_no_dc = power_spectrum.copy()
    dc_index = np.argmin(np.abs(f))
    power_spectrum_no_dc[dc_index] = 0  # zero out DC
    fundamental_idx = np.argmax(power_spectrum_no_dc)
    fundamental_freq = f[fundamental_idx]
    motif_dict[file] = int(abs(fundamental_freq))
    F0.append(int(abs(fundamental_freq)))
    #print(f"Fundamental frequency: {abs(fundamental_freq):.2f} Hz")

sorted_F0 = sorted(F0)

print(len(motif_dict))
list_motif = list(motif_dict.items())

median = np.median(sorted_F0)
collier = []
chardo = []

for x in list_motif:
    if x[1] <= median:
        collier.append(x)
    else:
        chardo.append(x)

print("Predicted Chardo")
for x in chardo:
    print(x)

print("Predicted Collier")
for x in collier:
    print(x)


