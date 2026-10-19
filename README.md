# Traditional ASR

This project experiments with **traditional automatic speech recognition (ASR)**
for isolated spoken digits. It uses short-time Fourier analysis, Mel-frequency
features, and Gaussian-mixture hidden Markov models (GMM-HMMs) rather than a
neural network.

The main workflow is:

1. Place Free Spoken Digit Dataset (FSDD)-style recordings in
   `data/recordings/`.
2. Inspect the recordings and their speaker/digit distribution.
3. Visualise waveform, spectrogram, log-Mel spectrogram, or MFCC features.
4. Train and evaluate a per-digit GMM-HMM.
5. Optionally compare it with the single-network GMM-HMM implementation.

## Dataset

The scripts expect mono WAV files in:

```text
data/recordings/
```

Filenames must follow this convention:

```text
<digit>_<speaker>_<index>.wav
```

For example:

```text
7_jackson_32.wav
```

The filename is used to obtain both the digit label and speaker name. The
repository includes utilities and notes related to the
[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).
FSDD recordings are commonly sampled at 8 kHz and contain 50 recordings of
each digit per speaker, but the feature extraction scripts use the sample rate
stored in each file.

## Requirements

- Python 3.10 or newer is recommended.
- A working Matplotlib backend is required for scripts that display plots.
- WAV recordings in `data/recordings/`.

The main analysis and recognition scripts use:

- `librosa`
- `numpy`
- `scipy`
- `soundfile`
- `matplotlib`
- `hmmlearn`
- `scikit-learn`

The optional Hub upload script additionally uses `hub`, `tqdm`, and the
dataset helper under `data/utils/`.

## Setup

Run all commands from the repository root.

### 1. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Install the dependencies used by the main pipeline:

```bash
python -m pip install --upgrade pip
python -m pip install librosa numpy scipy soundfile matplotlib hmmlearn scikit-learn
```

For the optional dataset upload workflow:

```bash
python -m pip install hub tqdm pillow
```

The historical dependency list in `data/pip_requirements.txt` belongs to the
original FSDD data utilities and contains very old pinned versions. It is not
recommended for the current analysis scripts.

### 3. Add or download recordings

Copy FSDD-compatible `.wav` files into `data/recordings/`. Verify that the
directory contains files such as `0_jackson_0.wav` before running the scripts.

### 4. Verify the environment and data

```bash
python check_setup.py
```

This prints installed library versions, counts the recordings, and loads one
sample file to report its sample rate, duration, and shape.

## Recommended workflow

### Inspect the dataset

```bash
python inspect_dataset.py
```

This reports the number of recordings, speakers, and examples per digit and
opens a waveform plot for one recording.

### Explore signal representations

Choose an audio file explicitly for the command-line visualisers:

```bash
python spectrogram.py data/recordings/7_jackson_32.wav
python logmelspectrogram.py data/recordings/7_jackson_32.wav
```

To inspect the first recording automatically:

```bash
python mfcc.py
python dft_mel.py
```

These scripts display plots with Matplotlib, so the command may wait until the
plot window is closed.

### Prepare and inspect a speaker split

```bash
python prepare_dataset.py
```

This extracts 39-dimensional MFCC, delta, and delta-delta feature sequences
and creates a speaker-based train/test split in memory. Edit
`TEST_SPEAKER` in the script before running it if the default speaker is not
present in the dataset.

### Train the per-digit GMM-HMM recogniser

```bash
python trainhmmgmm.py
```

This trains one left-to-right GMM-HMM for each digit, using all speakers except
`TEST_SPEAKER` for training. It then predicts the held-out speaker's digits,
prints accuracy and a classification report, and displays a confusion matrix.
Update `TEST_SPEAKER` in the script to a speaker that exists in your data.

### Train the single-network recogniser

```bash
python singlenetwork_hmm.py
```

This trains separate digit models and assembles their learned parameters into
one 30-state network (10 digits × 3 states). It scores each digit path through
the unified network and evaluates the held-out speaker. As above, update
`TEST_SPEAKER` when necessary.

## Python scripts

### Core analysis and recognition

| Script | Purpose |
| --- | --- |
| `check_setup.py` | Checks the required Python libraries, confirms that `data/recordings/` exists, counts WAV files, and loads a sample recording. |
| `inspect_dataset.py` | Summarises digit/speaker counts and plots a waveform from one recording. |
| `prepare_dataset.py` | Extracts 39-dimensional MFCC + delta + delta-delta sequences and creates a speaker-based train/test split for experimentation. |
| `spectrogram.py` | Plots a waveform and linear-frequency magnitude spectrogram in decibels for a supplied audio file. |
| `logmelspectrogram.py` | Plots a waveform, linear-frequency magnitude spectrogram, and 26-band log-Mel spectrogram. |
| `mfcc.py` | Extracts 13 MFCCs from the first recording and visualises the coefficient sequence. |
| `dft_mel.py` | Demonstrates the feature-extraction steps manually: windowing, FFT/DFT, power spectrum, Mel filterbank construction, and Mel energies. |
| `trainhmmgmm.py` | Trains one 4-state, 3-mixture GMM-HMM per digit and evaluates predictions on a held-out speaker. |
| `singlenetwork_hmm.py` | Builds a single 30-state GMM-HMM from per-digit models and scores constrained digit paths through that network. |

### Data and legacy utilities

These scripts support collecting, transforming, or publishing data. They are
not required for the main feature-extraction and recognition workflow.

| Script | Purpose |
| --- | --- |
| `data/acquire_data/say_numbers_prompt.py` | Prints a timed, shuffled sequence of digits for recording in Audacity. The original utility assumes an 8 kHz recording setup. |
| `data/acquire_data/split_and_label_numbers.py` | Splits one long recording into labelled digit WAV files and trims silence. Edit the recording path, speaker name, timing, and output directory before use. This file contains legacy Python 2-style output syntax and may need porting before it runs on Python 3. |
| `data/utils/trimmer.py` | Provides the original silence-trimming dependency used by the FSDD utilities. |
| `data/utils/spectogramer.py` | Converts WAV files to fixed-size PNG spectrograms. The filename is historically misspelled `spectogramer.py`. |
| `data/utils/train-test-split.py` | Copies spectrograms numbered 0–4 to a test directory and 5–49 to a training directory. Its relative paths assume it is run from the expected legacy data directory. |
| `data/utils/fsdd.py` | Provides the original FSDD helper for reading and generating dataset examples. |
| `data/upload_to_hub.py` | Uploads spectrograms, audio, labels, and speaker names to an Activeloop Hub dataset. Configure the local directories and dataset URI before running it. |
| `data/metadata.py` | Stores speaker metadata such as gender, accent, and language. |

## Feature-extraction settings

The core scripts use short-time analysis with settings close to standard
speech-processing practice:

- 25 ms analysis window
- 10 ms frame hop
- Hamming window
- 512-point FFT
- 26 Mel bands
- 13 static MFCCs
- 13 first-order deltas and 13 second-order deltas in the HMM scripts

The resulting HMM observation vector has 39 values per frame. Features are
standardised using training-speaker frames only before model fitting.

## Troubleshooting

### `No WAV files found`

Run the command from the repository root and check that recordings are located
directly under `data/recordings/`, not only in a nested directory.

### `TEST_SPEAKER not found`

Run `python inspect_dataset.py` or `python prepare_dataset.py`, copy one of the
printed speaker names, and assign it to `TEST_SPEAKER` in the training script.

### Plot windows do not appear

Run the script in a desktop environment with a supported Matplotlib backend.
For remote or headless environments, configure a non-interactive backend and
modify the plotting scripts to save figures instead of calling `plt.show()`.

### Audio format errors

Convert source recordings to valid mono WAV files. Check the sample rate and
channel count with `check_setup.py`; stereo files are averaged to mono by the
feature-extraction scripts.

## License and attribution

The included dataset documentation identifies FSDD as available under the
[Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/).
If you redistribute recordings or derived data, follow the dataset's licence
and attribution requirements.
