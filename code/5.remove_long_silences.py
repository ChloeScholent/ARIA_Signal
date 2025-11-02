import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

folder = "dynamically_filtered_dataset"
output_folder = "trimmed_dataset"
os.makedirs(output_folder, exist_ok=True)

top_db = 18                   # Silence detection threshold (dB)
silence_threshold_sec = 0.5   # Remove silences longer than this (seconds)

for file in os.listdir(folder):
    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(folder, file)
    y, sr = librosa.load(file_path, sr=None)

    # Detect non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)

    # Derive silent intervals
    silent_intervals = []
    prev_end = 0
    for start, end in intervals:
        if start > prev_end:
            silent_intervals.append((prev_end, start))
        prev_end = end

    # ✅ Add final silence if it reaches the end
    if prev_end < len(y):
        silent_intervals.append((prev_end, len(y)))

    # Convert threshold to samples
    silence_threshold_samples = int(silence_threshold_sec * sr)

    segments = []
    removed_regions = []
    prev_end = 0

    for start, end in silent_intervals:
        silence_len = end - start
        # Append the previous sound segment
        segments.append(y[prev_end:start])

        # Keep short silences, remove long ones
        if silence_len < silence_threshold_samples:
            segments.append(y[start:end])
        else:
            removed_regions.append((start / sr, end / sr))

        prev_end = end

    # ✅ Concatenate final result
    y_trimmed = np.concatenate(segments) if segments else np.array([])

    # Save output
    out_path = os.path.join(output_folder, file)
    sf.write(out_path, y_trimmed, sr)

    # --- Visualization ---
    plt.figure(figsize=(14, 6))

    # Original waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    for s, e in removed_regions:
        plt.axvspan(s, e, color='red', alpha=0.3)
    plt.title(f"Original waveform with removed silences marked ({file})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Trimmed waveform
    plt.subplot(2, 1, 2)
    if len(y_trimmed) > 0:
        librosa.display.waveshow(y_trimmed, sr=sr, alpha=0.8, color='green')
    plt.title("After removing long silences")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    print(f"Processed {file}:")
    print(f"  Removed silences > {silence_threshold_sec}s")
    print(f"  Original length: {len(y)/sr:.2f}s → Trimmed length: {len(y_trimmed)/sr:.2f}s\n")
