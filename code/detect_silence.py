import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from statistics import median
from pprint import pprint as print

folder = "dynamically_filtered_dataset"
mean_silence_size = []
file_silence_size = []
minim = []

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)

    if not file.endswith(".wav"):
        continue

    y, sr = librosa.load(file_path, sr=None)

    # Non-silent intervals
    intervals = librosa.effects.split(y, top_db=20)

    # Derive silent intervals
    silent_intervals = []
    prev_end = 0
    for start, end in intervals:
        if start > prev_end:
            silent_intervals.append((prev_end, start))
        prev_end = end

    # Compute silence lengths in samples
    silence_samples = []
    for interval in silent_intervals:
        silence_len = abs(interval[1] - interval[0])
        if silence_len < 5500:
            silence_samples.append(silence_len)
        
    if len(silence_samples) == 0:
        continue  # skip empty

    minim.append((len(silence_samples), file_path))
 
    # # Plot
    # plt.figure(figsize=(12, 4))
    # librosa.display.waveshow(y, sr=sr, alpha=0.7)
    # for s, e in silent_intervals:
    #     plt.axvspan(s / sr, e / sr, color='gray', alpha=0.4)
    # plt.title(f"Detected Silences in Birdsong ({file})")
    # plt.xlabel("Time (s)")
    # plt.show()

sorted_minim = sorted(minim)
print("\nFiles with most silent sequences:")
print(sorted_minim[20:])
