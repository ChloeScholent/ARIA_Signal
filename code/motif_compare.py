import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
import time

# ================================================================
# Optimized distance profile (Mueen’s algorithm)
# ================================================================
def fast_distance_profile_EUC(x, p):
    N = len(x)
    Np = len(p)
    nfft = N + Np

    c2 = np.cumsum(np.concatenate(([0], x**2)))
    cp = np.sum(p**2)

    p_flip = np.zeros(nfft)
    p_flip[:Np] = np.flip(p)

    X = np.fft.fft(x, nfft)
    P = np.fft.fft(p_flip, nfft)
    r = np.real(np.fft.ifft(X * P))
    r = r[Np-1:N+Np-1]

    d = np.sqrt(np.maximum(
        c2[Np:] - c2[:-Np] + cp - 2 * r[:N - Np + 1],
        0
    ))
    return d


# ================================================================
# Main matching loop
# ================================================================
motif_folder = "motif_csv_0.35/"
folder = "wav_resampled/"
arg = []
best_motif_paths = []

motif_list = sorted(os.listdir(motif_folder))

for file in sorted(os.listdir(folder)):
    file_path = os.path.join(folder, file)
    print(f"\nProcessing signal: {file} ...")

    # Load and normalize audio
    _, x1 = wavfile.read(file_path)
    x1 = x1.astype(np.float32)
    x1 = (x1 - np.mean(x1)) / np.std(x1)

    distances = []
    for motif in motif_list:
        motif_path = os.path.join(motif_folder, motif)
        p = np.loadtxt(motif_path, delimiter=",").astype(np.float32)
        p = (p - np.mean(p)) / np.std(p)

        d = fast_distance_profile_EUC(x1, p)
        distances.append(np.min(d))

    best_idx = np.argmin(distances)
    arg.append(best_idx)
    best_motif_paths.append(os.path.join(motif_folder, motif_list[best_idx]))

# ================================================================
# Results
# ================================================================
print("\n=== Best motif identified for each signal ===")
for i, idx in enumerate(arg):
    signal_name = sorted(os.listdir(folder))[i]
    motif_name = motif_list[idx]
    print(f"Signal: {signal_name} --> Best motif: {motif_name}")

unique_indices = np.unique(arg)
unique_motifs = [motif_list[i] for i in unique_indices]
unique_paths = [os.path.join(motif_folder, m) for m in unique_motifs]

print("\n=== Unique motifs selected ===")
for m, path in zip(unique_motifs, unique_paths):
    print(f"{m} --> {path}")

# (Optional) keep the mappings for later
signal_names = sorted(os.listdir(folder))
signal_to_motif = dict(zip(signal_names, [motif_list[i] for i in arg]))
