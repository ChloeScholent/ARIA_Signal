import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ================================================================
# SETTINGS
# ================================================================
motif_folder = ""
n_clusters = 2  # number of classes

# ================================================================
# FEATURE EXTRACTION
# ================================================================
features = []
motif_names = []

for motif in sorted(os.listdir(motif_folder)):
    if not motif.lower().endswith(".csv"):
        continue

    motif_path = os.path.join(motif_folder, motif)
    p = np.loadtxt(motif_path, delimiter=",")

    # Flatten in case the motif is multi-dimensional (e.g., stereo or 2D)
    p = np.ravel(p).astype(np.float32)

    # Normalize motif
    p = (p - np.mean(p)) / np.std(p)

    # Compute features — simple but effective
    f_mean = np.mean(p)
    f_std = np.std(p)
    f_max = np.max(p)
    f_min = np.min(p)
    f_skew = np.mean((p - np.mean(p))**3) / (np.std(p)**3 + 1e-8)
    f_kurt = np.mean((p - np.mean(p))**4) / (np.std(p)**4 + 1e-8)

    feature_vec = [f_mean, f_std, f_max, f_min, f_skew, f_kurt]
    features.append(feature_vec)
    motif_names.append(motif)

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
print("\n=== K-means Clustering Results (2 Classes) ===")
for name, label in zip(motif_names, labels):
    print(f"Motif: {name} --> Cluster {label}")

# Group results by cluster
cluster_0 = [m for m, l in zip(motif_names, labels) if l == 0]
cluster_1 = [m for m, l in zip(motif_names, labels) if l == 1]

print("\n=== Cluster 0 Motifs ===")
for m in cluster_0:
    print(f"  {m}")

print("\n=== Cluster 1 Motifs ===")
for m in cluster_1:
    print(f"  {m}")


pca = PCA(n_components=2)
X_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='coolwarm', s=100)
for i, name in enumerate(motif_names):
    plt.text(X_pca[i,0]+0.02, X_pca[i,1], name, fontsize=8)
plt.title("K-means Clustering of Motifs (2 Classes)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
