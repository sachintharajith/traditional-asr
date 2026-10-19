
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load a spoken-digit recording
# --------------------------------------------------
DATA_DIR = Path("data/recordings")
audio_files = sorted(DATA_DIR.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError(f"No WAV files found in {DATA_DIR.resolve()}")

audio_file = audio_files[0]
audio, sr = sf.read(audio_file)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

print("File:", audio_file.name)
print("Sample rate:", sr, "Hz")
print("Duration:", len(audio) / sr, "seconds")

# --------------------------------------------------
# 2. Extract one 25 ms frame
# --------------------------------------------------
frame_length = int(0.025 * sr)
hop_length = int(0.010 * sr)
n_fft = 512

# Select a frame starting at 0.1 seconds.
start = int(0.1 * sr)
frame = audio[start:start + frame_length]

# Handle unusually short recordings.
if len(frame) < frame_length:
    frame = np.pad(frame, (0, frame_length - len(frame)))

# --------------------------------------------------
# 3. Apply a Hamming window
# --------------------------------------------------
window = np.hamming(frame_length)
windowed_frame = frame * window

# --------------------------------------------------
# 4. Calculate the DFT using the FFT algorithm
# --------------------------------------------------
# Zero-padding to 512 points improves frequency-grid
# resolution, but does not add new signal information.
spectrum = np.fft.rfft(windowed_frame, n=n_fft)

# Power spectrum
power_spectrum = np.abs(spectrum) ** 2

frequencies = np.fft.rfftfreq(n_fft, d=1 / sr)

print("\nFrame length:", frame_length, "samples")
print("FFT size:", n_fft)
print("Number of frequency bins:", len(power_spectrum))

# --------------------------------------------------
# 5. Define Hz <-> Mel conversion functions
# --------------------------------------------------
def hz_to_mel(hz):
    return 2595 * np.log10(1 + np.asarray(hz) / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (np.asarray(mel) / 2595) - 1)


# --------------------------------------------------
# 6. Build triangular Mel filterbanks manually
# --------------------------------------------------
n_mels = 26
f_min = 0
f_max = sr / 2

mel_points = np.linspace(
    hz_to_mel(f_min),
    hz_to_mel(f_max),
    n_mels + 2
)

hz_points = mel_to_hz(mel_points)

# Map filter frequencies to FFT-bin indices.
bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
bins = np.clip(bins, 0, n_fft // 2)

filterbank = np.zeros((n_mels, n_fft // 2 + 1))

for m in range(1, n_mels + 1):
    left = bins[m - 1]
    center = bins[m]
    right = bins[m + 1]

    # Rising slope
    if center > left:
        for k in range(left, center):
            filterbank[m - 1, k] = (
                (k - left) / (center - left)
            )

    # Falling slope
    if right > center:
        for k in range(center, right):
            filterbank[m - 1, k] = (
                (right - k) / (right - center)
            )

# --------------------------------------------------
# 7. Apply filters to the power spectrum
# --------------------------------------------------
mel_energies = filterbank @ power_spectrum

# Log energies are commonly used before the DCT.
log_mel_energies = np.log(np.maximum(mel_energies, 1e-10))

print("\nFilterbank shape:", filterbank.shape)
print("Mel energy vector shape:", mel_energies.shape)

# --------------------------------------------------
# 8. Visualise the results
# --------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 13))

# Frame in the time domain
time = np.arange(frame_length) / sr * 1000
axes[0].plot(time, windowed_frame)
axes[0].set_title("Hamming-windowed speech frame")
axes[0].set_xlabel("Time (milliseconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

# Power spectrum
axes[1].plot(frequencies, power_spectrum)
axes[1].set_title("Power spectrum from the FFT")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power")
axes[1].grid(True)

# Triangular Mel filters
for m in range(n_mels):
    axes[2].plot(frequencies, filterbank[m])

axes[2].set_title("26 triangular Mel filterbanks")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Filter weight")
axes[2].set_xlim(0, sr / 2)
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Plot the resulting Mel-band energies separately.
plt.figure(figsize=(10, 4))
plt.plot(range(1, n_mels + 1), log_mel_energies, marker="o")
plt.title("Log Mel-band energies for one speech frame")
plt.xlabel("Mel filter index")
plt.ylabel("Log energy")
plt.grid(True)
plt.tight_layout()
plt.show()
