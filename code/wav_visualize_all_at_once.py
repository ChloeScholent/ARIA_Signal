import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


folder_path = "wav_resampled"
files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])


signals = []
sample_rates = []

for file in files:
    sr, data = wavfile.read(os.path.join(folder_path, file))
    sample_rates.append(sr)
    signals.append(data)


num_signals = len(signals)
rows = 8
cols = 5

fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
axes = axes.flatten()

for i, signal in enumerate(signals):
    t = np.arange(len(signal)) / sample_rates[i]  # time axis in seconds
    axes[i].plot(t, signal)
    axes[i].set_title(f"Signal {i+1}", fontsize=10)
    axes[i].tick_params(labelsize=8)

# Hide unused subplots
for j in range(i+1, rows*cols):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


