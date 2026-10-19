
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf

DATA_DIR = Path("data/recordings")

# Change these after checking the speakers in your dataset.
TEST_SPEAKER = "george"

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 80  # 10 ms at 8 kHz
WIN_LENGTH = 200  # 25 ms at 8 kHz


def extract_features(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # librosa's MFCC uses a Mel power spectrogram internally.
    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hamming",
        n_mels=26,
    )

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    # Shape: (frames, 39 features)
    return np.concatenate(
        [mfcc, delta, delta_delta], axis=0
    ).T


files = sorted(DATA_DIR.glob("*.wav"))

if not files:
    raise FileNotFoundError(f"No WAV files found in {DATA_DIR.resolve()}")

dataset = []
speakers = set()

for path in files:
    parts = path.stem.split("_")
    digit = int(parts[0])
    speaker = parts[1]
    speakers.add(speaker)

    features = extract_features(path)
    dataset.append({
        "path": path,
        "digit": digit,
        "speaker": speaker,
        "features": features,
    })

print("Speakers found:", sorted(speakers))

if TEST_SPEAKER == "the_speaker_name_here":
    raise ValueError(
        "Set TEST_SPEAKER to one speaker printed above, "
        "then rerun this script."
    )

if TEST_SPEAKER not in speakers:
    raise ValueError(f"Unknown test speaker: {TEST_SPEAKER}")

train_data = [
    item for item in dataset
    if item["speaker"] != TEST_SPEAKER
]

test_data = [
    item for item in dataset
    if item["speaker"] == TEST_SPEAKER
]

print(f"\nTraining recordings: {len(train_data)}")
print(f"Testing recordings:  {len(test_data)}")

print("\nExample feature sequence:")
example = train_data[0]
print("File:", example["path"].name)
print("Digit:", example["digit"])
print("Feature shape:", example["features"].shape)
print("(frames, features) =", example["features"].shape)
