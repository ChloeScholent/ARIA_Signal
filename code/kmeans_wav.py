import numpy as np
import os
from scipy.io import wavfile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import librosa

# ================================================================
# SETTINGS
# ================================================================
folder = "dynamically_filtered_dataset/"
n_clusters = 2  # we want two groups

# ================================================================
# FEATURE EXTRACTION
# ================================================================
features = []
file_names = []

for file in sorted(os.listdir(folder)):
    if not file.lower().endswith(".wav"):
        continue

    file_path = os.path.join(folder, file)
    print(f"Extracting features from {file} ...")

    # Load audio
    x, sr = librosa.load(file_path, sr=32000)  # resample to 32kHz

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)

    # Compute summary statistics (mean and std across time)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Concatenate mean and std into a single feature vector
    feature_vec = np.concatenate((mfcc_mean, mfcc_std))

    features.append(feature_vec)
    file_names.append(file)

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
    print(f"Signal: {fname} --> Cluster {label}")

# Optional: analyze cluster composition
cluster_0 = [f for f, l in zip(file_names, labels) if l == 0]
cluster_1 = [f for f, l in zip(file_names, labels) if l == 1]

print("\n=== Cluster 0 ===")
for f in cluster_0:
    print(f"  {f}")

print("\n=== Cluster 1 ===")
for f in cluster_1:
    print(f"  {f}")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='coolwarm', s=100)
for i, fname in enumerate(file_names):
    plt.text(X_pca[i,0]+0.02, X_pca[i,1], fname, fontsize=8)
plt.title("K-means Clustering of Birdsong Signals")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


