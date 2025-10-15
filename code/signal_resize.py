import os
import numpy as np
import soundfile as sf
from scipy.signal import stft

# --- Parameters ---
input_folder = "dynamically_filtered_dataset/"
output_folder = "trimmed_audio/"
min_freq = 3500     # Hz
max_freq = 6000     # Hz
frame_length = 1024
hop_length = 512
energy_threshold = 1e-2  # Skip very quiet frames

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".wav")):
        continue

    print(f"Processing {filename} ...")
    filepath = os.path.join(input_folder, filename)
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # Convert to mono

    # --- Compute STFT ---
    freqs, times, Zxx = stft(y, fs=sr, nperseg=frame_length, noverlap=frame_length - hop_length)
    magnitudes = np.abs(Zxx)

    # --- Dominant frequency per frame ---
    dominant_freqs = freqs[np.argmax(magnitudes, axis=0)]
    energies = np.sum(magnitudes, axis=0)

    # --- Keep frames in the desired frequency range and with sufficient energy ---
    keep_mask = (dominant_freqs >= min_freq) & (dominant_freqs <= max_freq) & (energies > energy_threshold)

    # --- Convert frames to time intervals ---
    frame_times = times
    segments = []
    start = None
    for i, keep in enumerate(keep_mask):
        if keep and start is None:
            start = frame_times[i]
        elif not keep and start is not None:
            segments.append((start, frame_times[i]))
            start = None
    if start is not None:
        segments.append((start, frame_times[-1] + (frame_length / sr)))

    # --- Concatenate valid segments ---
    trimmed_audio = np.concatenate([
        y[int(start * sr): int(end * sr)]
        for start, end in segments
    ]) if segments else np.array([], dtype=np.float32)

    # --- Save result ---
    if len(trimmed_audio) > 0:
        out_path = os.path.join(output_folder, filename)
        sf.write(out_path, trimmed_audio, sr)
        print(f"Saved trimmed file to: {out_path}")
    else:
        print(f"No valid segments found in {filename}")
