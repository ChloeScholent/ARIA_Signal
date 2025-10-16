import os
import numpy as np
import soundfile as sf

# --- Parameters ---
input_folder = "dynamically_filtered_dataset/"
output_folder = "trimmed_audio/"
duration_to_save = 5.0     # seconds
energy_threshold = 0.02    # adjust depending on audio volume
frame_size = 1024          # number of samples per analysis frame

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".wav", ".flac", ".ogg")):
        continue

    filepath = os.path.join(input_folder, filename)
    print(f"Processing {filename} ...")

    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # convert to mono

    # --- Compute short-time energy ---
    num_frames = len(y) // frame_size
    energies = np.array([
        np.mean(np.square(y[i*frame_size:(i+1)*frame_size]))
        for i in range(num_frames)
    ])

    # --- Find first frame above threshold ---
    idx = np.argmax(energies > energy_threshold)
    if energies[idx] <= energy_threshold:
        print(f"No energetic segment found in {filename}")
        continue

    start_time = idx * frame_size / sr
    end_time = start_time + duration_to_save
    print(f"Detected energy at {start_time:.2f}s → saving 5s segment")

    # --- Extract and save ---
    start_sample = int(start_time * sr)
    end_sample = int(min(end_time * sr, len(y)))
    trimmed = y[start_sample:end_sample]

    out_path = os.path.join(output_folder, filename)
    sf.write(out_path, trimmed, sr)
    print(f"Saved: {out_path}")
