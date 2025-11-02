import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import shutil

in_folder = "dataset/"
out_48000 = "trimmed_audio/48000"
out_44100 = "trimmed_audio/44100"
out_32000 = "trimmed_audio/32000"
os.makedirs(out_48000, exist_ok=True)
os.makedirs(out_44100, exist_ok=True)
os.makedirs(out_32000, exist_ok=True)

other = []

for file in os.listdir(in_folder):
    if not file.lower().endswith(".wav"):
        continue

    file_path = os.path.join(in_folder, file)
    name, ext = os.path.splitext(file)

    # Read sample rate (Fs)
    sample_rate, audio_data = wavfile.read(file_path)

    # Copy instead of move
    if sample_rate == 48000:
        out_path = os.path.join(out_48000, file)
        shutil.copy2(file_path, out_path)
    elif sample_rate == 44100:
        out_path = os.path.join(out_44100, file)
        shutil.copy2(file_path, out_path)
    elif sample_rate == 32000:
        out_path = os.path.join(out_32000, file)
        shutil.copy2(file_path, out_path)
    else:
        other.append(file_path)

print(f"✅ Files organized by sample rate. Other sample rates: {len(other)} files.\n")


def resample_folder(in_folder, out_folder, target_rate=32000):
    os.makedirs(out_folder, exist_ok=True)

    for file in sorted(os.listdir(in_folder)):
        if not file.lower().endswith(".wav"):
            continue

        in_path = os.path.join(in_folder, file)
        out_path = os.path.join(out_folder, file)

        Fs, audio = wavfile.read(in_path)
        print(f"🎧 {file} — original Fs={Fs} Hz")

        audio = audio.astype(np.float32)

        num_target_samples = int(len(audio) * target_rate / Fs)
        resampled = signal.resample(audio, num_target_samples)

        resampled /= np.max(np.abs(resampled) + 1e-12)

        resampled_int16 = np.int16(resampled * 32767)

        wavfile.write(out_path, target_rate, resampled_int16)
        print(f"✅ Saved resampled file → {out_path} ({target_rate} Hz)\n")


downsampled_out = "downsampled_dataset/"
os.makedirs(downsampled_out, exist_ok=True)

resample_folder("trimmed_audio/48000", downsampled_out, target_rate=32000)
resample_folder("trimmed_audio/44100", downsampled_out, target_rate=32000)
resample_folder("trimmed_audio/32000", downsampled_out, target_rate=32000)

print("🎉 All files resampled successfully!")
