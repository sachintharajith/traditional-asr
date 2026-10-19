from pathlib import Path
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.special import logsumexp
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
TEST_SPEAKER = "jackson"

N_MFCC = 13
N_STATES_PER_DIGIT = 4
N_MIXTURES = 3
N_DIGITS = 10
N_STATES = N_STATES_PER_DIGIT * N_DIGITS

N_FFT = 512
N_MELS = 26


# --------------------------------------------------
# 2. Extract 13 static MFCC features
# --------------------------------------------------
def extract_features(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")

    audio = audio.astype(np.float32)

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

    # First-order temporal derivatives
    delta = librosa.feature.delta(mfcc)

    # Second-order temporal derivatives
    delta_delta = librosa.feature.delta(
        mfcc,
        order=2,
    )

    # Stack 13 MFCC + 13 delta + 13 delta-delta
    # Shape changes from (39, frames) to (frames, 39)
    features = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0,
    ).T

    if not np.isfinite(features).all():
        raise ValueError(f"Invalid features in {path}")

    return features


# --------------------------------------------------
# 3. Load recordings and split by speaker
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

    records.append({
        "path": path,
        "digit": digit,
        "speaker": speaker,
        "features": extract_features(path),
    })

    speakers.add(speaker)

print("Speakers:", sorted(speakers))

if TEST_SPEAKER not in speakers:
    raise ValueError(
        f"Speaker '{TEST_SPEAKER}' not found. "
        f"Available speakers: {sorted(speakers)}"
    )

train_records = [
    r for r in records if r["speaker"] != TEST_SPEAKER
]
test_records = [
    r for r in records if r["speaker"] == TEST_SPEAKER
]

if not train_records or not test_records:
    raise ValueError("Training or test split is empty.")

print("Training recordings:", len(train_records))
print("Test recordings:", len(test_records))
print(
    "Training examples per digit:",
    dict(sorted(Counter(
        r["digit"] for r in train_records
    ).items())),
)


# --------------------------------------------------
# 4. Standardize using training data only
# --------------------------------------------------
scaler = StandardScaler()

scaler.fit(np.concatenate(
    [r["features"] for r in train_records],
    axis=0,
))

for record in train_records + test_records:
    record["features"] = scaler.transform(
        record["features"]
    )

print("Feature dimension:", N_MFCC)


# --------------------------------------------------
# 5. Train each labelled digit path
# --------------------------------------------------
digit_models = {}

for digit in range(N_DIGITS):
    digit_records = [
        r for r in train_records if r["digit"] == digit
    ]

    if not digit_records:
        raise ValueError(
            f"No training recordings for digit {digit}"
        )

    sequences = [r["features"] for r in digit_records]
    X = np.concatenate(sequences, axis=0)
    lengths = [len(seq) for seq in sequences]

    # Three-state left-to-right topology.
    startprob = np.zeros(N_STATES_PER_DIGIT)
    startprob[0] = 1.0

    transmat = np.zeros(
        (N_STATES_PER_DIGIT, N_STATES_PER_DIGIT)
    )

    for state in range(N_STATES_PER_DIGIT - 1):
        transmat[state, state] = 0.6
        transmat[state, state + 1] = 0.4

    transmat[-1, -1] = 1.0

    model = GMMHMM(
        n_components=N_STATES_PER_DIGIT,
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

    # Check learned parameters before assembling the network.
    for name, values in [
        ("start probabilities", model.startprob_),
        ("transition probabilities", model.transmat_),
        ("mixture weights", model.weights_),
        ("means", model.means_),
        ("covariances", model.covars_),
    ]:
        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid {name} in digit {digit} model"
            )

    digit_models[digit] = model

    print(
        f"Trained digit {digit}: "
        f"{len(digit_records)} recordings"
    )


# --------------------------------------------------
# 6. Assemble ONE 30-state GMM-HMM network
# --------------------------------------------------
# Each digit occupies its own three-state block.
# The blocks are connected to a shared entry distribution
# but cannot transition into one another.

network = GMMHMM(
    n_components=N_STATES,
    n_mix=N_MIXTURES,
    covariance_type="diag",
    random_state=42,
)

network.startprob_ = np.zeros(N_STATES)
network.transmat_ = np.zeros((N_STATES, N_STATES))

network.weights_ = np.concatenate([
    digit_models[d].weights_
    for d in range(N_DIGITS)
], axis=0)

network.means_ = np.concatenate([
    digit_models[d].means_
    for d in range(N_DIGITS)
], axis=0)

network.covars_ = np.concatenate([
    digit_models[d].covars_
    for d in range(N_DIGITS)
], axis=0)

for digit in range(N_DIGITS):
    start = digit * N_STATES_PER_DIGIT
    stop = start + N_STATES_PER_DIGIT

    # Equal prior probability for each digit path.
    network.startprob_[start] = 1.0 / N_DIGITS

    # Copy the trained transition block into the network.
    network.transmat_[start:stop, start:stop] = (
        digit_models[digit].transmat_
    )

print("\nUnified network assembled.")
print("Total states:", network.n_components)
print("Gaussian mixtures per state:", N_MIXTURES)


# --------------------------------------------------
# 7. Score one path through the unified network
# --------------------------------------------------
def score_digit_path(log_emissions, digit):
    """
    Forward algorithm restricted to one digit's
    three-state path in the unified network.
    """
    start = digit * N_STATES_PER_DIGIT
    stop = start + N_STATES_PER_DIGIT

    emissions = log_emissions[:, start:stop]
    transitions = network.transmat_[start:stop, start:stop]

    # Start in the first state of this digit's path.
    log_transitions = np.full_like(
        transitions, -np.inf, dtype=float
    )
    positive = transitions > 0
    log_transitions[positive] = np.log(
        transitions[positive]
    )

    T = emissions.shape[0]
    alpha = np.full(
        (T, N_STATES_PER_DIGIT), -np.inf
    )

    alpha[0, 0] = emissions[0, 0]

    for t in range(1, T):
        for state in range(N_STATES_PER_DIGIT):
            alpha[t, state] = emissions[t, state] + logsumexp(
                alpha[t - 1] + log_transitions[:, state]
            )

    return logsumexp(alpha[-1])


# --------------------------------------------------
# 8. Recognize test recordings
# --------------------------------------------------
y_true = []
y_pred = []

for record in test_records:
    features = record["features"]

    # Compute emissions using the unified network's
    # 30 state distributions.
    log_emissions = network._compute_log_likelihood(
        features
    )

    scores = {
        digit: score_digit_path(log_emissions, digit)
        for digit in range(N_DIGITS)
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
# 9. Evaluate
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(N_DIGITS)),
    zero_division=0,
))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=list(range(N_DIGITS)),
    display_labels=list(range(N_DIGITS)),
    cmap="Blues",
    values_format="d",
)

plt.title(
    f"Single-Network GMM-HMM — Test Speaker: {TEST_SPEAKER}"
)
plt.tight_layout()
plt.show()