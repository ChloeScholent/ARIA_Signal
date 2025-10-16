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
out_folder = "motif_extraction/"
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

        # === Plot ===
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))

        # 1️⃣ Original signal with motif highlighted
        axs[0].plot(x, color="gray", alpha=0.7)
        axs[0].axvspan(motif_index, motif_index + w, color="red", alpha=0.3, label="Most significant motif")
        axs[0].set_title(f"{name} - Original Signal with Motif Highlighted")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()

        # 2️⃣ Extracted motif (zoomed view)
        axs[1].plot(motif, color="red")
        axs[1].set_title("Extracted Most Significant Motif")
        axs[1].set_ylabel("Amplitude")

        # 3️⃣ Matrix profile
        axs[2].plot(matrix_profile, color="black")
        axs[2].plot(motif_index, matrix_profile[motif_index], "*r")
        axs[2].set_title("Matrix Profile (Lowest Value = Strongest Motif)")
        axs[2].set_xlabel("Sample Index")
        axs[2].set_ylabel("Profile Value")

        plt.tight_layout()
        output_path = os.path.join(out_folder, f"{L}{name}_motif.pdf")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"📄 Saved plot: {output_path}")
