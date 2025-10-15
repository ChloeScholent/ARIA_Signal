import numpy as np
import stumpy
from scipy.io import wavfile, signal
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import time, os

folder = "dynamically_filtered_dataset/"
out_folder = "motif_extraction/"
os.makedirs(out_folder, exist_ok=True)

max_duration = 20  # seconds (trim longer files)
downsample_factor = 0.5  # optional, 2× faster
threshold_gpu = 200_000  # samples threshold for GPU use

for file in os.listdir(folder):
    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(folder, file)
    name, ext = os.path.splitext(file)
    print(f"\nProcessing {file}...")

    # === Load audio ===
    sample_rate, audio_data = wavfile.read(file_path)
    Fs = sample_rate

    # Convert stereo to mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Trim long audio
    if len(audio_data) > max_duration * Fs:
        audio_data = audio_data[:int(max_duration * Fs)]

    # Downsample (optional)
    if downsample_factor < 1.0:
        audio_data = signal.resample(audio_data, int(len(audio_data) * downsample_factor))
        Fs *= downsample_factor

    # Convert to float64 (required by STUMPY)
    x = audio_data.astype(np.float64)

    # === Parameters ===
    L = 0.14
    w = int(L * Fs)

    print(f"Window size: {w} samples ({L}s), signal length: {len(x)} samples")

    # === Compute matrix profile ===
    start = time.time()
    if len(x) < threshold_gpu:
        mp = stumpy.stump(x, m=w)
        method = "CPU"
    else:
        mp = stumpy.gpu_stump(x, m=w)
        method = "GPU"

    elapsed = time.time() - start
    print(f"✅ {method} matrix profile computed in {elapsed:.2f} seconds")

    # === Extract motifs ===
    matrix_profile = mp[:, 0]
    i = np.argmin(matrix_profile)
    j = int(mp[i, 1])
    motif1 = x[i:i+w]
    motif2 = x[j:j+w]

    print(f"Motif pair at indices {i} and {j}")

    # === Plot and save ===
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[0].plot(x, color="gray", alpha=0.6)
    axs[0].axvspan(i, i+w, color="red", alpha=0.3)
    axs[0].axvspan(j, j+w, color="blue", alpha=0.3)
    axs[0].set_title(f"{name} - Input Signal with Motif Locations")

    axs[1].plot(matrix_profile, color="black")
    axs[1].plot(i, matrix_profile[i], "*r")
    axs[1].set_title("Matrix Profile")

    axs[2].plot(motif1, "r", label="Motif 1")
    axs[2].plot(motif2, "b", label="Motif 2")
    axs[2].legend()
    axs[2].set_title("Motif Pair")

    plt.tight_layout()
    plt.savefig(f"{out_folder}/{name}.pdf")
    plt.close(fig)
