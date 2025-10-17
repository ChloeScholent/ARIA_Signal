import numpy as np
import stumpy
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # faster for SSH
import matplotlib.pyplot as plt
import os, time

# === Paths ===
folder = "trimmed_audio/"
out_folder = "motif_csv/"
os.makedirs(out_folder, exist_ok=True)

# === Processing parameters ===
max_duration = 20        # seconds (trim long audio)
downsample_factor = 0.5  # optional, speeds up 2×
threshold_gpu = 200_000  # switch between CPU/GPU
L_list = [.1, .15, .17, .2, .25, .3]   # motif window length (seconds)
for L in L_list:
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder, file)
        name, _ = os.path.splitext(file)
        print(f"\n🎧 Processing: {file}")

        # === Load and preprocess ===
        sample_rate, audio_data = wavfile.read(file_path)
        Fs = sample_rate

        # Trim to avoid massive FFTs
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
        print(f" - Window size: {w} samples ({L:.2f}s)")
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

        # === Find most significant motif ===
        matrix_profile = mp[:, 0]
        motif_index = np.argmin(matrix_profile)
        motif = x[motif_index:motif_index + w]

        print(f"🎯 Most significant motif starts at index {motif_index}")

        # === Prepare output CSV path ===
        csv_path = os.path.join(out_folder, f"{L}_{name}_motifs.csv")

        # === Save to CSV ===
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Header row
            header = ["motif_length", "motif_index"] + [f"sample_{j}" for j in range(w)]
            writer.writerow(header)

            # Write each motif as a row
            for rank, i in enumerate(motif_indices, start=1):
                row = [rank, i] + list(x[i:i + w])
                writer.writerow(row)

        print(f"💾 Saved top {top_k} motifs to {csv_path}")
