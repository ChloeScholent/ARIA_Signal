import numpy as np
import matplotlib.pyplot as plt
import os


folder = "segmented_motif/"


for file in sorted(os.listdir(folder)):
    file_path = os.path.join(folder, file)

    # Load the motif from your saved CSV
    motif = np.loadtxt(file_path, delimiter=",")
    print(type(motif), motif.shape, type(motif[0]))
    # Print some info
    print(f"Motif loaded with {len(motif)} samples")
    print(file)
    # print("First 10 samples:", motif[:10])

    plt.figure(figsize=(8, 3))
    plt.plot(motif)
    plt.title("Extracted Motif Waveform")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()



#Single motif visualization

# file = "motif_csv/0.1dynamic_mono_XC467936_motif.csv"

# motif = np.loadtxt(file, delimiter=",")

# Print some info
# print(f"Motif loaded with {len(motif)} samples")

# plt.figure(figsize=(8, 3))
# plt.plot(motif)
# plt.title("Extracted Motif Waveform")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()
