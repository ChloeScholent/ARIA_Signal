import numpy as np
import stumpy
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import os, time
import csv

# === Paths ===
folder = "short_dataset/"
out_folder = "motif_0.5/"
os.makedirs(out_folder, exist_ok=True)

# === Processing parameters ===
max_duration = 20        # seconds
downsample_factor = 0.5  # optional, speeds up 2×
threshold_gpu = 200_000  # switch between CPU/GPU
L = .5                # motif window length (seconds)

# === Loop over all .wav files ===
for file in os.listdir(folder):
    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(folder, file)
    name, _ = os.path.splitext(file)
    print(f"\n🎧 Processing: {file}")

    # === Load and preprocess ===
    sample_rate, audio_data = wavfile.read(file_path)
    Fs = sample_rate

    # Trim to max_duration
    if len(audio_data) > max_duration * Fs:
        audio_data = audio_data[:int(max_duration * Fs)]

    # Optional downsampling
    if downsample_factor < 1.0:
        audio_data = signal.resample(audio_data, int(len(audio_data) * downsample_factor))
        Fs *= downsample_factor

    # Convert to float64 for STUMPY
    x = audio_data.astype(np.float64)

    # === Define window size ===
    w = int(L * Fs)
    print(f" - Window size: {w} samples")
    print(f" - Signal length: {len(x)} samples")

    # === Compute Matrix Profile ===
    start = time.time()
    if len(x) < threshold_gpu:
        mp = stumpy.stump(x, m=w)
        method = "CPU"
    else:
        mp = stumpy.gpu_stump(x, m=w)
        method = "GPU"
    elapsed = time.time() - start
    print(f"✅ {method} matrix profile computed in {elapsed:.2f} s")

    # === Extract best motif ===
    matrix_profile = mp[:, 0]
    best_index = np.argmin(matrix_profile)
    best_motif = x[best_index:best_index + w]
    print(f"🎯 Best motif starts at index {best_index}")

    # === Save best motif to CSV ===
    csv_file = os.path.join(out_folder, f"{name}_best_motif.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(best_motif)  # one row with the motif
    print(f"💾 Saved best motif to {csv_file}")

print("\n✅ All signals processed, best motif per file saved.")
