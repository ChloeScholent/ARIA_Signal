import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
wav_folder = "dynamically_filtered_dataset"       # <-- your WAV files folder
motif_folder = "motif_csv_0.35"   # <-- folder with 40 motif CSVs
threshold_method = "mean"             # "median", "mean", or manual number

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def normalize(x):
    x = x.astype(np.float32)
    return (x - np.mean(x)) / np.std(x)

# -----------------------------
# LOAD WAV FILES
# -----------------------------
signals = []
file_names = []
for fname in os.listdir(wav_folder):
    if fname.endswith(".wav"):
        sr, data = wavfile.read(os.path.join(wav_folder, fname))
        if data.ndim > 1:  # convert stereo to mono
            data = np.mean(data, axis=1)
        signals.append(normalize(data))
        file_names.append(fname)

# -----------------------------
# LOOP OVER MOTIFS
# -----------------------------
motif_files = [f for f in os.listdir(motif_folder) if f.endswith(".csv")]

for motif_fname in motif_files:
    # Load motif
    motif_path = os.path.join(motif_folder, motif_fname)
    motif = np.loadtxt(motif_path, delimiter=",")
    motif = normalize(motif)
    
    # Compute similarity
    similarities = []
    for s in signals:
        corr = correlate(s, motif, mode='valid')
        max_corr = np.max(np.abs(corr))
        similarities.append(max_corr)
    
    # Set threshold
    if threshold_method == "median":
        threshold = np.median(similarities)
    elif threshold_method == "mean":
        threshold = np.mean(similarities)
    else:
        threshold = float(threshold_method)
    
    # Classify
    labels = ["Class1" if sim >= threshold else "Class2" for sim in similarities]
    
    # Print results
    print(f"\n=== Motif: {motif_fname} | Threshold: {threshold:.2f} ===")
    for fname, label, sim in zip(file_names, labels, similarities):
        print(f"{fname}: {label} (similarity={sim:.2f})")
    
    # Sort signals by file name
    sorted_indices = np.argsort(file_names)  # get sorted indices
    sorted_names = [file_names[i] for i in sorted_indices]
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(24, 6))  # wide figure
    plt.bar(sorted_names, sorted_similarities, color=['green' if l=="Class1" else 'red' for l in sorted_labels])
    plt.axhline(y=threshold, color='blue', linestyle='--', label='Threshold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Max Correlation")
    plt.title(f"Similarity to motif {motif_fname}")
    plt.legend()
    plt.tight_layout()
    plt.show()
