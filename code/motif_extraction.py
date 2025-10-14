import numpy as np
import stumpy
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time

# === Load and preprocess audio ===
sample_rate, audio_data = wavfile.read("dynamically_filtered_dataset/dynamic_mono_XC1029284.wav")
Fs = sample_rate

# Convert to mono if stereo
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)

# Normalize audio (float64 required by STUMPY)
x = audio_data.astype(np.float64)
x = x - np.mean(x)
x = x / np.std(x)

# === Define parameters ===
L = 0.14              # window length in seconds
w = int(L * Fs)       # convert to samples

print(f"Running GPU matrix profile (window = {w} samples)...")

# === Compute matrix profile on GPU ===
start = time.time()
mp = stumpy.gpu_stump(x, m=w)
elapsed = time.time() - start
print(f"✅ Done in {elapsed:.2f} seconds")

# === Extract results ===
matrix_profile = mp[:, 0]
profile_index = np.argmin(matrix_profile)
print(f"Minimum matrix profile index: {profile_index}")

# === Identify motif pair (two most similar subsequences) ===
i = np.argmin(mp[:, 0])  # motif 1 start index
j = int(mp[i, 1])        # motif 2 start index (nearest neighbor)
print(f"Motif pair found at indices {i} and {j}")

motif1 = x[i:i+w]
motif2 = x[j:j+w]

# === Plot results ===
plt.figure("Matrix Profile and Motifs", figsize=(12, 8))

# 1️⃣ Original signal
plt.subplot(3, 1, 1)
plt.plot(x, color="gray", alpha=0.6)
plt.axvspan(i, i+w, color="red", alpha=0.3, label="Motif 1")
plt.axvspan(j, j+w, color="blue", alpha=0.3, label="Motif 2")
plt.title("Input Audio Signal with Motif Locations")
plt.ylabel("Amplitude")
plt.legend()

# 2️⃣ Matrix profile
plt.subplot(3, 1, 2)
plt.plot(matrix_profile, color="black")
plt.plot(i, matrix_profile[i], "*r")
plt.title("Matrix Profile (GPU Accelerated with STUMPY)")
plt.ylabel("Matrix Profile Value")

# 3️⃣ Extracted motif pair
plt.subplot(3, 1, 3)
plt.plot(motif1, "r", label="Motif 1")
plt.plot(motif2, "b", label="Motif 2")
plt.title("Extracted Motif Pair (Most Similar Segments)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
