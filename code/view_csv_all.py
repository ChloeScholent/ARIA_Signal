import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
motif_folder = "motif_0.5/"  # folder with CSVs

# === Collect all motif files ===
motif_files = [f for f in os.listdir(motif_folder) if f.endswith(".csv")]
signals = []
sample_rates = []
# === Plot each motif individually ===
for file in motif_files:
    path = os.path.join(motif_folder, file)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        row = next(reader)  # only one row per CSV
        motif = np.array([float(x) for x in row])
        sr=32000
        sample_rates.append(sr)
        signals.append(motif)


num_signals = len(signals)
rows = 8
cols = 5

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
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









