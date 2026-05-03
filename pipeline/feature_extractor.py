"""
Feature extraction: Mel spectrogram (128 bins) + MFCC (40 dim x 94 frames).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
from utils.config_handler import pipeline_conf


def extract_features(waveform: np.ndarray, sample_rate: int) -> dict:
    """Extract Mel and MFCC features for one 3s segment.

    Args:
        waveform: 1D numpy array (48000 samples for 3s @ 16kHz).
        sample_rate: Sample rate in Hz.

    Returns:
        dict with keys:
          - mel: np.ndarray shape (1, 128, T_mel) — ready for Conformer branch
          - mfcc: np.ndarray shape (1, 1, 40, 94) — ready for CNN branch
    """
    conf = pipeline_conf["pipeline"]
    n_fft = conf["n_fft"]
    hop_length = conf["hop_length"]
    n_mels = conf["mel_bins"]
    n_mfcc = conf["mfcc_dim"]

    # Mel spectrogram (center=False for exact frame count)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate,
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        center=conf["center"],
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Add batch dim: (1, 128, T_mel)
    mel_out = mel_db[np.newaxis, :, :].astype(np.float32)

    # MFCC (center=False matching mel)
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sample_rate,
        n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        center=conf["center"],
    )

    # MFCC shape: (1, 1, 40, 94) — batch=1, channel=1
    mfcc_out = mfcc[np.newaxis, np.newaxis, :, :].astype(np.float32)

    return {
        "mel": mel_out,
        "mfcc": mfcc_out,
    }
