import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import shutil

folder = "trimmed_audio/"
out_1 = "trimmed_audio/48000"
out_2 = 'trimmed_audio/44100'
out_3 = "trimmed_audio/32000"

big = []
medium = []
small = []
other = []

for file in os.listdir(folder):
    if not file.endswith(".wav"):
        continue
    file_path = os.path.join(folder, file)
    name, ext = os.path.splitext(file)
    sample_rate, audio_data = wavfile.read(file_path)
    Fs = sample_rate
    if Fs == 48000:
        big.append(file_path)
        out_path = os.path.join(out_1, file)
        shutil.move(file_path, out_path)
    elif Fs == 44100:
        medium.append(file_path)
        out_path = os.path.join(out_2, file)
        shutil.move(file_path, out_path)
    elif Fs == 32000:
        small.append(file_path)
        out_path = os.path.join(out_3, file)
        shutil.move(file_path, out_path)
    else:
        other.append(file_path)
print()
print(big)
print()
print(small)