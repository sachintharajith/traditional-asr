
from pathlib import Path
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
DATA_DIR = Path("data/recordings")

# Replace with an actual speaker name from your dataset.
TEST_SPEAKER = "jackson"

N_MFCC = 13
N_STATES = 4
N_MIXTURES = 3
N_FFT = 512
N_MELS = 26

# --------------------------------------------------
# 2. Extract MFCC + delta + delta-delta features
# --------------------------------------------------
def extract_features(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")

    audio = audio.astype(np.float32)

    # Use the actual sample rate to configure the frame length.
    win_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=hop_length,
        win_length=win_length,
        window="hamming",
        n_mels=N_MELS,
    )

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    # (39, frames) -> (frames, 39)
    return np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0,
    ).T


# --------------------------------------------------
# 3. Load dataset and split by speaker
# --------------------------------------------------
files = sorted(DATA_DIR.glob("*.wav"))

if not files:
    raise FileNotFoundError(
        f"No WAV recordings found in {DATA_DIR.resolve()}"
    )

records = []
speakers = set()

for path in files:
    parts = path.stem.split("_")
    digit = int(parts[0])
    speaker = parts[1]

    features = extract_features(path)

    records.append({
        "path": path,
        "digit": digit,
        "speaker": speaker,
        "features": features,
    })
    speakers.add(speaker)

print("Speakers:", sorted(speakers))

if TEST_SPEAKER not in speakers:
    raise ValueError(
        f"TEST_SPEAKER='{TEST_SPEAKER}' not found. "
        f"Choose one of: {sorted(speakers)}"
    )

train_records = [
    r for r in records if r["speaker"] != TEST_SPEAKER
]
test_records = [
    r for r in records if r["speaker"] == TEST_SPEAKER
]

if not train_records or not test_records:
    raise ValueError("Training or test split is empty.")

print("\nTraining recordings:", len(train_records))
print("Testing recordings:", len(test_records))

print(
    "Training examples per digit:",
    dict(sorted(Counter(r["digit"] for r in train_records).items()))
)

# --------------------------------------------------
# 4. Standardize features using training data only
# --------------------------------------------------
scaler = StandardScaler()

# Every frame is a 39-dimensional observation.
all_train_frames = np.concatenate(
    [r["features"] for r in train_records],
    axis=0,
)
scaler.fit(all_train_frames)

for record in train_records + test_records:
    record["features"] = scaler.transform(record["features"])

print("\nFeature dimension:", train_records[0]["features"].shape[1])

# --------------------------------------------------
# 5. Train one GMM-HMM per digit
# --------------------------------------------------
models = {}

for digit in range(10):
    digit_records = [
        r for r in train_records if r["digit"] == digit
    ]

    if not digit_records:
        raise ValueError(f"No training recordings for digit {digit}")

    sequences = [r["features"] for r in digit_records]

    # hmmlearn expects one concatenated matrix plus sequence lengths.
    X = np.concatenate(sequences, axis=0)
    lengths = [len(seq) for seq in sequences]

    # Initialize a left-to-right HMM topology.
    startprob = np.zeros(N_STATES)
    startprob[0] = 1.0

    transmat = np.zeros((N_STATES, N_STATES))

    for state in range(N_STATES - 1):
        transmat[state, state] = 0.6
        transmat[state, state + 1] = 0.4

    # Final state can only remain in itself.
    transmat[-1, -1] = 1.0

    model = GMMHMM(
        n_components=N_STATES,
        n_mix=N_MIXTURES,
        covariance_type="diag",
        min_covar=1e-3,
        n_iter=100,
        tol=0.01,
        random_state=42,
        verbose=False,
        init_params="mcw",
        params="stmcw",
    )

    model.startprob_ = startprob
    model.transmat_ = transmat

    model.fit(X, lengths)
    models[digit] = model

    print(
        f"Trained digit {digit}: "
        f"{len(digit_records)} recordings, "
        f"{N_STATES} states, {N_MIXTURES} mixtures/state"
    )

# --------------------------------------------------
# 6. Recognize each test recording
# --------------------------------------------------
y_true = []
y_pred = []

for record in test_records:
    features = record["features"]

    scores = {
        digit: model.score(features)
        for digit, model in models.items()
    }

    predicted_digit = max(scores, key=scores.get)

    y_true.append(record["digit"])
    y_pred.append(predicted_digit)

    print(
        f"{record['path'].name:30s} "
        f"Actual={record['digit']} "
        f"Predicted={predicted_digit}"
    )

# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(10)),
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=list(range(10)),
    display_labels=list(range(10)),
    cmap="Blues",
    values_format="d",
)
plt.title(f"GMM-HMM Digit Recognition — Test Speaker: {TEST_SPEAKER}")
plt.tight_layout()
plt.show()
