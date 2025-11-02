import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ================================================================
# SETTINGS
# ================================================================
motif_folder = "motif_0.15/"
n_clusters = 2  # we want two groups
sr = 32000      # sampling rate for MFCC
mfcc_n = 13     # number of MFCCs
min_length = 2048  # minimum length for MFCC extraction

# ================================================================
# FEATURE EXTRACTION
# ================================================================
features = []
file_names = []

for motif_file in sorted(os.listdir(motif_folder)):
    if not motif_file.lower().endswith(".csv"):
        continue

    motif_path = os.path.join(motif_folder, motif_file)
    x = np.loadtxt(motif_path, delimiter=",")

    # Normalize motif
    x = (x - np.mean(x)) / np.std(x)

    # Pad or truncate motif to min_length to ensure MFCC works
    if len(x) < min_length:
        x = np.pad(x, (0, min_length - len(x)), 'constant')
    else:
        x = x[:min_length]

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=mfcc_n)

    # Summary statistics: mean and std across time frames
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Concatenate mean and std to form feature vector
    feature_vec = np.concatenate((mfcc_mean, mfcc_std))

    features.append(feature_vec)
    file_names.append(motif_file)

features = np.array(features)

# ================================================================
# STANDARDIZE FEATURES
# ================================================================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ================================================================
# K-MEANS CLUSTERING
# ================================================================
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
labels = kmeans.fit_predict(features_scaled)

# ================================================================
# RESULTS
# ================================================================
print("\n=== K-means Clustering Results ===")
for fname, label in zip(file_names, labels):
    print(f"Motif: {fname} --> Cluster {label}")

# Cluster composition
cluster_0 = [f for f, l in zip(file_names, labels) if l == 0]
cluster_1 = [f for f, l in zip(file_names, labels) if l == 1]

print("\n=== Cluster 0 ===")
for f in cluster_0:
    print(f"  {f}")

print("\n=== Cluster 1 ===")
for f in cluster_1:
    print(f"  {f}")

# ================================================================
# OPTIONAL: PCA VISUALIZATION
# ================================================================
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='coolwarm', s=100)
for i, fname in enumerate(file_names):
    plt.text(features_pca[i, 0] + 0.02, features_pca[i, 1], fname, fontsize=8)
plt.title("K-means Clustering of Motif Signals (PCA 2D Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
