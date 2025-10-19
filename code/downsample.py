import os
import numpy as np
from scipy.io import wavfile
from scipy import signal

# === Paths ===
in_folder = "trimmed_audio/48000"
out_folder = "wav_resampled/"
os.makedirs(out_folder, exist_ok=True)

# === Target sample rate ===
target_rate = 32000

# === Loop through all wav files ===
for file in sorted(os.listdir(in_folder)):
    if not file.lower().endswith(".wav"):
        continue

    in_path = os.path.join(in_folder, file)
    out_path = os.path.join(out_folder, file)

    # --- Read original file ---
    Fs, audio = wavfile.read(in_path)
    print(f"🎧 {file} — original Fs={Fs} Hz")

    # Convert to float for processing
    audio = audio.astype(np.float32)

    # --- Resample ---
    num_target_samples = int(len(audio) * target_rate / Fs)
    resampled = signal.resample(audio, num_target_samples)

    # Normalize to avoid clipping (optional)
    resampled /= np.max(np.abs(resampled) + 1e-12)

    # Convert back to int16 for saving
    resampled_int16 = np.int16(resampled * 32767)

    # --- Save ---
    wavfile.write(out_path, target_rate, resampled_int16)
    print(f"✅ Saved resampled file → {out_path} ({target_rate} Hz)\n")

print("🎉 All files resampled successfully!")
