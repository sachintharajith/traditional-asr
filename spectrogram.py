
import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def visualise_spectrogram(audio_path):
    # Load audio, preserving its original sample rate
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # STFT configuration
    n_fft = 512
    win_length = int(0.025 * sr)  # 25 ms window
    hop_length = int(0.010 * sr)  # 10 ms hop

    # Compute the Short-Time Fourier Transform
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hamming",
        center=True,
    )

    # Convert complex STFT values to magnitude
    magnitude = np.abs(stft)

    # Convert magnitude to decibels
    magnitude_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
    )

    # Time and frequency axes
    times = librosa.frames_to_time(
        np.arange(magnitude.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft,
    )

    # Plot waveform and spectrogram
    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 9),
        sharex=False,
    )

    duration = len(audio) / sr
    audio_times = np.arange(len(audio)) / sr

    axes[0].plot(audio_times, audio)
    axes[0].set_title("Audio waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True)

    image = axes[1].pcolormesh(
        times,
        frequencies,
        magnitude_db,
        shading="auto",
        cmap="magma",
        vmin=-80,
        vmax=0,
    )

    axes[1].set_title("Magnitude spectrogram (dB)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_ylim(0, sr / 2)

    fig.colorbar(
        image,
        ax=axes[1],
        label="Magnitude (dB relative to maximum)",
    )

    fig.suptitle(
        f"{Path(audio_path).name} | "
        f"Sample rate: {sr} Hz | "
        f"Duration: {duration:.2f} s"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise an audio waveform and spectrogram."
    )

    parser.add_argument(
        "audio_file",
        help="Path to a WAV or another format supported by librosa",
    )

    args = parser.parse_args()
    visualise_spectrogram(args.audio_file)
