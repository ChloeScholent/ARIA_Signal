import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import os

folder = "dataset/"
file_path = "dynamically_filtered_dataset/dynamic_mono_XC1029284.wav"
# for file in os.listdir(folder):
#     file_path = os.path.join(folder, file)
#     name, ext = os.path.splitext(file)
sample_rate, audio_data = wavfile.read(file_path)
Fs = sample_rate
x = audio_data
N = np.size(x)
t=np.arange(N)/Fs

#DATA INFO
print(f"Sampling Frequency : {Fs} Hz")
print(f"Number of samples : {N}")
print(f'Length of the signal : {N/Fs:.2f}s')
print('\n')


#SIGNAL VISUALIZATION
plt.figure("x[n] as a function of the time t[n]")
plt.plot(t,x)
plt.xlim((0,(N-1)/Fs))
plt.xlabel('Time (seconds)')
plt.title('$x[n]$')
plt.show()




    
