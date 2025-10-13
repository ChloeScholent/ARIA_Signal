import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy import signal
import os


folder = "filtered_dataset/"

# for file in os.listdir(folder):
#     file_path = os.path.join(folder, file)
#     name, ext = os.path.splitext(file)
sample_rate, audio_data = wavfile.read("filtered_dataset/filtered_mono_XC707455.wav")
Fs = sample_rate
x = audio_data
N = np.size(x)
t=np.arange(N)/Fs


f, t_spec, Sxx = signal.spectrogram(x, Fs)

# Plot
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()