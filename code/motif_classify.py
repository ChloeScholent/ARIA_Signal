import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from dtaidistance import dtw
from sklearn.cluster import AgglomerativeClustering


# Parameters
folder_path = "motif_csv/" 
resample_length = 1600

# Load signals
signals = []
file_names = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, file_name))
        signal = df.iloc[:, 0].values  # assuming signal is in first column
        signals.append(signal)
        file_names.append(file_name)

# ----------------------------
# Resample signals to same length
# ----------------------------
signals_resampled = [resample(sig, resample_length) for sig in signals]

# ----------------------------
# Compute DTW distance matrix
# ----------------------------
n = len(signals_resampled)
distance_matrix = np.zeros((n, n))

print("Computing DTW distances...")
for i in range(n):
    for j in range(i, n):
        dist = dtw.distance(signals_resampled[i], signals_resampled[j])
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

# ----------------------------
# Perform hierarchical clustering
# ----------------------------
clustering = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
labels = clustering.fit_predict(distance_matrix)

# ----------------------------
# Print cluster assignments
# ----------------------------
for file_name, label in zip(file_names, labels):
    print(f"{file_name} --> Cluster {label}")

# ----------------------------
# Plot signals colored by cluster
# ----------------------------
plt.figure(figsize=(12, 6))
colors = ['r', 'b']
for i, sig in enumerate(signals_resampled):
    plt.plot(sig, color=colors[labels[i]], alpha=0.7)
plt.title("Signals colored by cluster")
plt.xlabel("Resampled time")
plt.ylabel("Signal amplitude")
plt.show()
