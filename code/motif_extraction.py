import cupy as cp
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time

# GPU version of matrix profile
def matrix_profile(x, w):
    N = x.size
    d = cp.inf * cp.ones((N - w,))
    for i in range(N - w):
        x_ = x[i:i + w]
        c = fast_distance_profile_nEUC(x, x_)
        c[cp.maximum(0, int(i - w)):cp.minimum(N - w, int(i + w))] = cp.inf
        d = cp.minimum(d, c)
    return d

def fast_distance_profile_nEUC(x, p):
    c = cp.cumsum(cp.concatenate(([0], x)))
    c2 = cp.cumsum(cp.concatenate(([0], x)) ** 2)
    N = x.size
    Np = p.size

    std_p = cp.std(p)
    if std_p == 0:
        return cp.inf * cp.ones(N - Np)

    p_ = (p - cp.mean(p)) / std_p
    p__ = cp.zeros((N,))
    p__[0:Np] = cp.flip(p_)

    # FFT-based correlation on GPU
    r = cp.real(cp.fft.ifft(cp.multiply(cp.fft.fft(x), cp.fft.fft(p__))))
    vari = cp.sqrt(Np * (c2[Np:-1] - c2[:N - Np]) - (c[Np:-1] - c[:N - Np]) ** 2)
    d = cp.sqrt(cp.maximum(2 * Np * (1 - cp.divide(r[Np - 1:N - 1], vari)), 0))
    return d

# Load audio
sample_rate, audio_data = wavfile.read("dynamically_filtered_dataset/dynamic_mono_XC1029284.wav")
Fs = sample_rate

# Convert to mono if stereo
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)

# Transfer data to GPU
x = cp.asarray(audio_data.astype(cp.float32))
N = x.size

L = 0.14  # seconds
w = int(L * Fs)

print(f"Running matrix profile on GPU (window = {w} samples)...")
start = time.time()
m = matrix_profile(x, w)
cp.cuda.Stream.null.synchronize()  # wait for GPU to finish
print(f"✅ Done in {time.time() - start:.2f} s")

# Transfer result back to CPU for plotting
m_cpu = cp.asnumpy(m)
x_cpu = cp.asnumpy(x)

plt.figure("Matrix profile")
plt.subplot(2, 1, 1)
plt.plot(x_cpu)
plt.ylabel("Input signal")
plt.subplot(2, 1, 2)
plt.plot(m_cpu)
plt.ylabel("Matrix Profile")
ind = m_cpu.argmin()
plt.plot(ind, m_cpu[ind], "*r")
plt.legend(("Matrix profile", "Minimum matrix profile value"))
plt.show()
