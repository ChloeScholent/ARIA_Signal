from kmeans_leraton import K_means
import os
import numpy as np
from numpy.fft import rfft
from pprint import pprint as print


folder = "motif_0.3"
signals = []

for file in sorted(os.listdir(folder)):
    file_path = os.path.join(folder, file)
    motif = np.loadtxt(file_path, delimiter=",")
    signals.append(motif)

fft_signals =[]

for sig in signals:
    fft_signals.append(rfft(sig))

fft_signals = np.array(fft_signals)


centers, clusters_points, cluster_index = K_means(points=fft_signals, nb_clust=2)

print(cluster_index)

print(centers)