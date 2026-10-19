
from pathlib import Path

import librosa
import hmmlearn
import numpy as np
import sklearn

DATA_DIR = Path("data/recordings")

print("NumPy:", np.__version__)
print("Librosa:", librosa.__version__)
print("scikit-learn:", sklearn.__version__)
print("hmmlearn:", hmmlearn.__version__)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Recordings directory not found: {DATA_DIR.resolve()}"
    )

files = sorted(DATA_DIR.glob("*.wav"))

print("Total WAV recordings:", len(files))
print("First five recordings:")

for file in files[:5]:
    print(" ", file.name)

if not files:
    raise RuntimeError("No WAV recordings found.")

audio, sample_rate = librosa.load(files[0], sr=None, mono=True)

print("\nSample recording:", files[0].name)
print("Sample rate:", sample_rate, "Hz")
print("Duration:", round(len(audio) / sample_rate, 2), "seconds")
print("Audio shape:", audio.shape)
