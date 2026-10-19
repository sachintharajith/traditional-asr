
from pathlib import Path
from collections import Counter

import librosa
import matplotlib.pyplot as plt
import soundfile as sf

DATA_DIR = Path("data/recordings")

# Find all recordings
files = sorted(DATA_DIR.glob("*.wav"))

if not files:
    raise FileNotFoundError(f"No WAV files found in {DATA_DIR.resolve()}")

# Inspect digit and speaker distribution
digit_counts = Counter()
speaker_counts = Counter()

for file in files:
    parts = file.stem.split("_")

    digit = int(parts[0])
    speaker = parts[1]

    digit_counts[digit] += 1
    speaker_counts[speaker] += 1

print(f"Total recordings: {len(files)}")
print(f"Number of speakers: {len(speaker_counts)}")
print(f"Digits found: {sorted(digit_counts.keys())}")

print("\nRecordings per digit:")
for digit, count in sorted(digit_counts.items()):
    print(f"Digit {digit}: {count}")

print("\nRecordings per speaker:")
for speaker, count in sorted(speaker_counts.items()):
    print(f"{speaker}: {count}")

# Inspect one audio recording
sample_file = files[499]
audio, sample_rate = sf.read(sample_file)

print("\nSample recording:", sample_file.name)
print("Sample rate:", sample_rate, "Hz")
print("Duration:", round(len(audio) / sample_rate, 3), "seconds")
print("Audio shape:", audio.shape)

# Plot the audio waveform
time = range(len(audio))

plt.figure(figsize=(12, 4))
plt.plot([i / sample_rate for i in time], audio)
plt.title(f"Speech waveform: {sample_file.name}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
