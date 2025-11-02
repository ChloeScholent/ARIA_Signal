import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.decomposition import PCA
import librosa

# --- Paths ---
signals_dir = "short_dataset"
motifs_dir = "motif_0.3"

# --- Load signals ---
signals = []
for file in sorted(os.listdir(signals_dir)):
    if file.endswith(".wav"):
        y, sr = librosa.load(os.path.join(signals_dir, file), sr=None, mono=True)
        signals.append(y)
signals = np.array(signals, dtype=object)

# --- Load motifs ---
motifs = []
for file in sorted(os.listdir(motifs_dir)):
    if file.endswith(".csv"):
        motif = np.loadtxt(os.path.join(motifs_dir, file), delimiter=",")
        motifs.append(motif)


# --- Compute max correlations ---
n_signals = len(signals)
n_motifs = len(motifs)
corr_matrix = np.zeros((n_signals, n_motifs))

for i, sig in enumerate(signals):
    for j, motif in enumerate(motifs):
        corr = correlate(sig, motif, mode='valid')
        corr_matrix[i, j] = np.max(np.abs(corr))

# --- PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(corr_matrix)

# --- Plot PCA scatter (signals) ---
plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color="steelblue", label="Signals")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA of Max Correlations (Signals vs Motifs)")
plt.grid(True)
plt.legend()
plt.show()

# --- Correlation circle (motifs) ---
# Correlations between original variables (motifs) and PCs
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

plt.figure(figsize=(7, 7))
for i in range(n_motifs):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
              color='orange', alpha=0.7, head_width=0.03, length_includes_head=True)
    plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, f"Motif {i+1}", ha='center', va='center')

# Draw circle
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Correlation Circle of Motifs")
plt.axis('equal')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import librosa

# --- Assume you already have: ---
# signals (list of np.arrays)
# motifs (np.array)
# corr_matrix (max correlations matrix)

motif_idx = 24  # Motif 25 (0-indexed)
motif = motifs[motif_idx]

# Define a threshold: e.g., 80% of max correlation
threshold = 0.8 * np.max(corr_matrix[:, motif_idx])
strong_matches_idx = np.where(corr_matrix[:, motif_idx] >= threshold)[0]

print(f"Signals strongly matching Motif 25 (threshold={threshold:.2f}):")
print(strong_matches_idx + 1)  # +1 for human-readable indices

# --- Plot motif over each strong match ---
for sig_idx in strong_matches_idx:
    signal = signals[sig_idx]
    
    # Find best matching segment via cross-correlation
    corr = correlate(signal, motif, mode='valid')
    best_pos = np.argmax(np.abs(corr))
    
    # Extract segment
    matched_segment = signal[best_pos : best_pos + len(motif)]
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(matched_segment, label=f"Signal {sig_idx+1} segment")
    plt.plot(motif, label="Motif 25", alpha=0.7)
    plt.title(f"Motif 25 vs Signal {sig_idx+1}")
    plt.legend()
    plt.show()
