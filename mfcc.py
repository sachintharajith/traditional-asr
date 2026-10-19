
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data/recordings")

# Select one recording
audio_file = sorted(DATA_DIR.glob("*.wav"))[0]

# Load audio at its original sample rate
audio, sample_rate = librosa.load(
    audio_file,
    sr=None,
    mono=True
)

# Extract 13 MFCC coefficients per frame
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13,
    n_fft=512,
    win_length=int(0.025 * sample_rate),
    hop_length=int(0.010 * sample_rate),
    n_mels=26
)

# Shape: (13, number_of_frames)
print("Audio file:", audio_file.name)
print("Sample rate:", sample_rate)
print("MFCC shape:", mfcc.shape)

# Transpose to (number_of_frames, 13)
features = mfcc.T

print("Feature matrix shape:", features.shape)
print("\nFirst frame's 13 MFCC coefficients:")
print(np.round(features[0], 2))

# Visualise the MFCC sequence
plt.figure(figsize=(12, 5))
plt.imshow(
    mfcc,
    aspect="auto",
    origin="lower",
    interpolation="nearest"
)
plt.title(f"MFCC features: {audio_file.name}")
plt.xlabel("Frame index")
plt.ylabel("MFCC coefficient")
plt.colorbar(label="Coefficient value")
plt.tight_layout()
plt.show()
