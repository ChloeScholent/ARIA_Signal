import os
import numpy as np
import soundfile as sf

input_folder = "trimmed_dataset/"
output_folder = "short_dataset/"
target_duration = 5.0 

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    y, sr = sf.read(filepath)

    total_duration = len(y) / sr

    if total_duration < target_duration:

        print(f"⏩ {filename}: {total_duration:.2f}s < {target_duration}s → unchanged")
        trimmed = y
    else:

        target_samples = int(target_duration * sr)
        trimmed = y[:target_samples]
        print(f"✂️ {filename}: trimmed to {target_duration}s ({target_samples} samples)")

    out_path = os.path.join(output_folder, filename)
    sf.write(out_path, trimmed, sr)
    print(f"✅ Saved: {out_path}")

print("\n🎉 All files processed successfully!")
