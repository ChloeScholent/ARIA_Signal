import numpy as np

# Load the motif from your saved CSV
motif = np.loadtxt("motif_csv/dynamic_mono_XC1017470_L0.10s_motif.csv", delimiter=",")

# Print some info
print(f"Motif loaded with {len(motif)} samples")
print("First 10 samples:", motif[:10])


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
plt.plot(motif)
plt.title("Extracted Motif Waveform")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
