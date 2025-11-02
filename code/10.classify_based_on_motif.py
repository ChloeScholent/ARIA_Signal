import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.signal import correlate
from pprint import pprint as print

# -----------------------------
# CONFIGURATION
# -----------------------------
wav_folder = "short_dataset"       # <-- your WAV files folder
motif_folder = "motif_0.3"        # <-- folder with 40 motif CSVs
threshold_method = "median"        # "median", "mean", or manual number

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

for fname in sorted(os.listdir(wav_folder)):
    if fname.endswith(".wav"):
        sr, data = wavfile.read(os.path.join(wav_folder, fname))
        if data.ndim > 1:  # convert stereo to mono
            data = np.mean(data, axis=1)
        signals.append(normalize(data))
        file_names.append(fname)

# -----------------------------
# LOAD MOTIFS
# -----------------------------
motif_files = [f for f in sorted(os.listdir(motif_folder)) if f.endswith(".csv")]

# Initialize DataFrame to store classifications
classification_df = pd.DataFrame(index=file_names, columns=motif_files)

# -----------------------------
# COMPUTE SIMILARITIES & CLASSIFICATIONS
# -----------------------------
for motif_fname in motif_files:
    motif_path = os.path.join(motif_folder, motif_fname)
    motif = np.loadtxt(motif_path, delimiter=",")
    motif = normalize(motif)
    
    # Compute similarity for each signal
    similarities = []
    for s in signals:
        corr = correlate(s, motif, mode='valid')
        max_corr = np.max(np.abs(corr))
        similarities.append(max_corr)
    
    # Determine threshold
    if threshold_method == "median":
        threshold = np.median(similarities)
    elif threshold_method == "mean":
        threshold = np.mean(similarities)
    else:
        threshold = float(threshold_method)
    
    # Classify
    labels = ["Class1" if sim >= threshold else "Class2" for sim in similarities]
    
    # Save classifications in DataFrame
    classification_df[motif_fname] = labels

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print(classification_df)

# -----------------------------
# HEATMAP VISUALIZATION
# -----------------------------
# Convert Class1/Class2 to numbers for heatmap: Class1=1, Class2=0
heatmap_data = classification_df.replace({"Class1": 1, "Class2": 0})

plt.figure(figsize=(16, 8))
sns.heatmap(
    heatmap_data,
    cmap="viridis",
    cbar_kws={'label': 'Class (1=Class1, 0=Class2)'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title("Signal Classification per Motif")
plt.xlabel("Motifs")
plt.ylabel("Signals")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
