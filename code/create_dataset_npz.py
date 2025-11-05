import numpy as np
from scipy.io import wavfile
import os

in_folder = "dataset/"
out_file = "birdsong_dataset_mono.npz"

signals = []
sample_rates = []

# Load all WAV files
for filename in sorted(os.listdir(in_folder)):
    if not filename.lower().endswith(".wav"):
        continue  # skip non-wav files

    file_path = os.path.join(in_folder, filename)
    sr, data = wavfile.read(file_path)

    # Convert to mono if stereo
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    signals.append(data.astype(np.float32))
    sample_rates.append(sr)

# Sanity check
print(f"Loaded {len(signals)} signals.")
print("Sample rates:", set(sample_rates))

# Save all data in a single npz file
np.savez(
    out_file,
    signals=np.array(signals, dtype=object),  # store variable-length arrays
    sample_rates=np.array(sample_rates)
)

print(f"Saved dataset to {out_file}")
