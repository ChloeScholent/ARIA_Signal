
import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
import os

# Function to compute the centered Fast Fourier Transform (FFT)
def my_fft(x,Fs):
    N=np.size(x)
    #Computation of the FFT
    X=np.fft.fft(x)
    X=np.fft.fftshift(X)
    # Computation the frequency vector
    f=np.fft.fftfreq(N, d=1/Fs)
    f=np.fft.fftshift(f)
    return X,f

def band_pass_filter(x, fc, Fs):
    # fc is a tuple (low_cut, high_cut)
    wc = [f / (Fs/2) for f in fc]  # normalize each frequency
    b, a = signal.butter(4, wc, btype='bandpass')
    y = signal.filtfilt(b, a, x)
    return y

def dynamic_band_pass_filter(x, Fs, threshold_ratio=0.4, margin_hz=50, exclude_below=3500, exclude_above=6000):
    """
    Automatically choose fc based on FFT peak frequencies.
    threshold_ratio: ratio (0–1) of max magnitude to define strong frequency band.
    margin_hz: frequency margin to widen the band edges.
    """
    N = len(x)
    freqs = rfftfreq(N, 1/Fs)
    fft_mag = np.abs(rfft(x))

    valid_idx = (freqs >= exclude_below) & (freqs <= exclude_above)
    freqs = freqs[valid_idx]
    fft_mag = fft_mag[valid_idx]

    threshold = threshold_ratio * np.max(fft_mag)
    strong_freqs = freqs[fft_mag > threshold]
    
    if len(strong_freqs) == 0:
        print("No strong frequencies detected. Returning original signal.")
        return x, (0, Fs/2)

    f_low = max(0, np.min(strong_freqs) - margin_hz)
    f_high = min(Fs/2, np.max(strong_freqs) + margin_hz)
    fc = (f_low, f_high)
    
    print(f"Auto-selected band: {f_low:.1f} Hz – {f_high:.1f} Hz for {file}")
    
    y = band_pass_filter(x, fc, Fs)
    
    return y, fc


folder = "dataset/"
new_folder = "dynamically_filtered_dataset/"

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    name, ext = os.path.splitext(file)
    sample_rate, audio_data = wavfile.read(file_path)
    Fs = sample_rate
    x = audio_data
    N = np.size(x)
    t=np.arange(N)/Fs

    y, fc = dynamic_band_pass_filter(x, Fs, threshold_ratio=0.2, exclude_above=8000)
    
    y, fc = dynamic_band_pass_filter(y, Fs)

    # Plot comparison
    # plt.figure(figsize=(10, 5))
    # plt.subplot(2, 1, 1)
    # plt.magnitude_spectrum(x, Fs=Fs, color='gray', label='Original')
    # plt.title("Original spectrum")

    # plt.subplot(2, 1, 2)
    # plt.magnitude_spectrum(y, Fs=Fs, color='green', label='Filtered')
    # plt.title(f"Dynamic Band-Pass Filtered Signal\n(f_low={fc[0]:.1f} Hz, f_high={fc[1]:.1f} Hz)")
    # plt.tight_layout()
    # plt.show()

    #PLot filtered signal vs original signal
    # plt.figure("Band-pass filter - Time domain")
    # plt.plot(t,x)
    # plt.plot(t,y)
    # plt.xlim((0,(N-1)/Fs))
    # plt.xlabel('Time (seconds)')
    # plt.legend(('Original signal', 'Filtered signal'))
    # plt.show()

    output_file = f'{new_folder}dynamic_{name}{ext}'
    # Normalize to 16-bit range for audio
    y_norm = np.int16((y / np.max(np.abs(y))) * 32767)
    wavfile.write(output_file, Fs, y_norm)