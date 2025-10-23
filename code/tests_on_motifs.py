import numpy as np
import matplotlib.pyplot as plt
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
folder = "motif_csv/"
Fs = 32000  # Sampling frequency (Hz)
motif_dict = {}

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)

    # Load motif
    motif = np.loadtxt(file_path, delimiter=",")
    #print(f"\nMotif: {file} | Samples: {len(motif)}")

    # Compute FFT
    X, f = my_fft(motif, Fs)

    # Compute power spectrum
    power_spectrum = np.abs(X) ** 2

    # Ignore DC (0 Hz) and find the frequency with maximum power
    mid = len(f) // 2
    power_spectrum_no_dc = power_spectrum.copy()
    dc_index = np.argmin(np.abs(f))
    power_spectrum_no_dc[dc_index] = 0  # zero out DC
    fundamental_idx = np.argmax(power_spectrum_no_dc)
    fundamental_freq = f[fundamental_idx]
    motif_dict[int(abs(fundamental_freq))] = file
    F0.append(int(abs(fundamental_freq)))
    #print(f"Fundamental frequency: {abs(fundamental_freq):.2f} Hz")

sorted_F0 = sorted(F0)

for x, y in list(motif_dict.items()):
    for f in sorted_F0[20:]:
        if f == x:
            print(x, y)
            del motif_dict[abs(x)]
            sorted_F0.remove(f)
