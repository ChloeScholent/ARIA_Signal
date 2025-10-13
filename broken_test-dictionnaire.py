import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

# ===========================================================
#             GPU Dictionary Learning with CuPy
# ===========================================================

def create_frames(x, Nw, Nh, window=None):
    """Create overlapping frames from a 1D signal."""
    n_frames = 1 + (len(x) - Nw) // Nh  # cover full signal
    idx = cp.arange(Nw)[:, None] + cp.arange(0, n_frames * Nh, Nh)[None, :]
    X = x[idx]
    if window is not None:
        X = X * window[:, None]
    return X

def icreate_frames(X, Nh, window=None):
    """
    Reconstruct the signal from overlapping frames using overlap-add.
    Works on GPU (CuPy arrays).
    """
    Nw, Nd = X.shape
    N = (Nd - 1) * Nh + Nw
    y = cp.zeros(N, dtype=cp.float32)
    w = cp.zeros(N, dtype=cp.float32)

    for i in range(Nd):
        start = i * Nh
        end = start + Nw
        frame = X[:, i]
        if window is not None:
            frame = frame * window
        y[start:end] += frame
        w[start:end] += (window if window is not None else 1)

    w = cp.maximum(w, 1e-8)
    y /= w
    return y

def hard_threshold(C, K0):
    """Keep only the largest K0 values per column."""
    absC = cp.abs(C)
    thresh = cp.partition(absC, -K0, axis=0)[-K0, :]
    mask = absC >= thresh
    return C * mask

def sparse_coding_matrix(X, D, K0, n_iter=50):
    """Vectorized sparse coding using Hard Thresholding Gradient Descent."""
    gamma = 1 / (cp.linalg.norm(D, 2) ** 2)
    Z = cp.zeros((D.shape[1], X.shape[1]), dtype=cp.float32)
    for _ in range(n_iter):
        R = D @ Z - X
        C = Z - gamma * (D.T @ R)
        Z = hard_threshold(C, K0)
    return Z

def dictionary_learning_matrix(X, Z, D, n_iter=30):
    """Vectorized dictionary learning with column normalization."""
    gamma = 1 / (cp.linalg.norm(Z, 2) ** 2)
    for _ in range(n_iter):
        R = D @ Z - X
        D -= gamma * (R @ Z.T)
        D /= cp.sqrt(cp.sum(D**2, axis=0, keepdims=True)) + 1e-8
    return D


# ===========================================================
#                     Load audio
# ===========================================================
sample_rate, audio_data = wavfile.read("dataset/mono_XC77547.wav")
Fs = sample_rate
x_cpu = audio_data.astype(np.float32)
x = cp.asarray(x_cpu)
N = x.size
t = np.arange(N) / Fs

print(f"Sampling Frequency : {Fs} Hz")
print(f"Number of samples  : {N}")
print(f"Length of the signal : {N/Fs:.2f} s\n")

from scipy.signal import butter, filtfilt

# ===========================================================
#                     Load audio
# ===========================================================
sample_rate, audio_data = wavfile.read("dataset/mono_XC77547.wav")
Fs = sample_rate
x_cpu = audio_data.astype(np.float32)

import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, freqz

# ===========================================================
#                     Load audio
# ===========================================================
sample_rate, audio_data = wavfile.read("dataset/mono_XC77547.wav")
Fs = sample_rate
x_cpu = audio_data.astype(np.float32)
N = len(x_cpu)
t = np.arange(N) / Fs

print(f"Sampling Frequency : {Fs} Hz")
print(f"Number of samples  : {N}")
print(f"Length of the signal : {N/Fs:.2f} s\n")

# ===========================================================
#              Band-pass filter: 4–5 kHz
# ===========================================================
lowcut = 4000.0
highcut = 5000.0
order = 6  # 6th-order Butterworth filter

nyq = 0.5 * Fs
low = lowcut / nyq
high = highcut / nyq

# Design filter
b, a = butter(order, [low, high], btype='band')

# Frequency response plot (optional)
w, h = freqz(b, a, worN=8000)
plt.figure("Band-pass Filter Frequency Response", figsize=(6, 3))
plt.plot((Fs * 0.5 / np.pi) * w, abs(h))
plt.title(f"Butterworth Band-pass Filter ({lowcut:.0f}-{highcut:.0f} Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply zero-phase filtering
x_filt = filtfilt(b, a, x_cpu)

print(f"✅ Filtrage passe-bande appliqué entre {lowcut:.0f} Hz et {highcut:.0f} Hz")

# ===========================================================
#               Compare original vs filtered signal
# ===========================================================
plt.figure("Original vs Filtered Signal", figsize=(10, 4))
plt.plot(t, x_cpu, label="Original", alpha=0.5)
plt.plot(t, x_filt, label=f"Filtered ({lowcut:.0f}-{highcut:.0f} Hz)", linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Comparison: Original vs Band-pass Filtered Signal")
plt.tight_layout()
plt.show()

# ===========================================================
#           Continue with GPU dictionary learning
# ===========================================================
x = cp.asarray(x_filt)  # Send filtered signal to GPU


# ===========================================================
#                     Parameters
# ===========================================================
Nw = 512      # frame length (~12 ms)
Nh = 256      # hop length (50% overlap)
K = 64        # number of dictionary atoms
K0 = 8        # sparsity level
n_epochs = 100
batch_size = 2000
window = cp.hanning(Nw).astype(cp.float32)

# ===========================================================
#                    Prepare frames
# ===========================================================
X = create_frames(x, Nw, Nh, window)
Nw, Nd = X.shape
print(f"Created {Nd} frames of length {Nw}")

# Initialize random dictionary
cp.random.seed(0)
D = cp.random.randn(Nw, K, dtype=cp.float32)
D -= cp.mean(D, axis=0, keepdims=True)
D /= cp.sqrt(cp.sum(D**2, axis=0, keepdims=True)) + 1e-8

# ===========================================================
#                   Training loop (GPU)
# ===========================================================
print("\nStarting GPU dictionary learning...\n")
for epoch in range(n_epochs):
    perm = cp.random.permutation(Nd)
    X = X[:, perm]

    for start in range(0, Nd, batch_size):
        end = min(start + batch_size, Nd)
        X_batch = X[:, start:end]
        Z_batch = sparse_coding_matrix(X_batch, D, K0)
        D = dictionary_learning_matrix(X_batch, Z_batch, D)

    if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch + 1}/{n_epochs} completed")

print("\n✅ Training complete!\n")

# ===========================================================
#             Compute full activations with final D
# ===========================================================
print("Computing sparse codes for all frames...")
Z = sparse_coding_matrix(X, D, K0)

# ===========================================================
#              Reconstruct the signal from frames
# ===========================================================
print("Reconstructing full signal...")
X_bar = D @ Z
y = icreate_frames(X_bar, Nh, window)

# Convert to CPU for plotting & saving
x_cpu = cp.asnumpy(x)
y_cpu = cp.asnumpy(y)
t_cpu = np.arange(len(y_cpu)) / Fs

# ===========================================================
#              Align lengths for plotting
# ===========================================================
min_len = min(len(x_cpu), len(y_cpu))
x_cpu = x_cpu[:min_len]
y_cpu = y_cpu[:min_len]
t_cpu = t_cpu[:min_len]

# ===========================================================
#                      Plot Results
# ===========================================================
D_cpu = cp.asnumpy(D)
plt.figure("Learned Dictionary Atoms", figsize=(10, 8))
for k in range(K):
    plt.subplot(int(np.ceil(K/8)), 8, k + 1)
    plt.plot(D_cpu[:, k])
plt.suptitle("Learned Dictionary Atoms")
plt.tight_layout()
plt.show()

plt.figure("Signal Reconstruction", figsize=(12, 4))
plt.plot(t_cpu, x_cpu, label="Original signal", alpha=0.6)
plt.plot(t_cpu, y_cpu, label="Reconstructed signal", alpha=0.6)
plt.legend()
plt.xlabel("Time (s)")
plt.title("Full Signal Reconstruction from Learned Dictionary")
plt.tight_layout()
plt.show()

# ===========================================================
#                     Save reconstructed WAV
# ===========================================================
out_path = "reconstructed.wav"
wavfile.write(out_path, Fs, y_cpu.astype(np.int16))
print(f"✅ Reconstruction complete and saved to '{out_path}'")
