
import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def visualise_spectrogram(audio_path):
    # 1. Load audio at its original sample rate
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # 2. Analysis parameters
    n_fft = 512
    win_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)  # 10 ms
    n_mels = 26

    # 3. Compute the STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hamming",
        center=True,
    )

    # Linear-frequency magnitude spectrogram
    magnitude = np.abs(stft)

    magnitude_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
    )

    # 4. Compute the Mel power spectrogram
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hamming",
        center=True,
        n_mels=n_mels,
        fmin=0,
        fmax=sr / 2,
        power=2,
    )

    # Convert Mel power to a logarithmic dB scale
    log_mel_db = librosa.power_to_db(
        mel_power,
        ref=np.max,
    )

    # 5. Display all three plots
    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 13),
    )

    duration = len(audio) / sr
    audio_times = np.arange(len(audio)) / sr

    # Waveform
    axes[0].plot(audio_times, audio)
    axes[0].set_title("Audio Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True)

    # Linear-frequency spectrogram
    img1 = librosa.display.specshow(
        magnitude_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title("Magnitude Spectrogram (dB)")
    axes[1].set_ylim(0, sr / 2)
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    # Log-Mel spectrogram
    img2 = librosa.display.specshow(
        log_mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=0,
        fmax=sr / 2,
        ax=axes[2],
        cmap="magma",
    )
    axes[2].set_title(f"Log-Mel Spectrogram ({n_mels} Mel Bands)")
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")

    fig.suptitle(
        f"{Path(audio_path).name} | "
        f"Sample rate: {sr} Hz | Duration: {duration:.2f} s"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot waveform, magnitude spectrogram and log-Mel spectrogram."
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file",
    )

    args = parser.parse_args()
    visualise_spectrogram(args.audio_file)
