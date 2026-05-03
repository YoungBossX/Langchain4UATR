"""
Audio preprocessing: noise reduction, normalization, pre-emphasis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import noisereduce as nr
from utils.config_handler import pipeline_conf


def preprocess(waveform: np.ndarray, sample_rate: int) -> dict:
    """Apply noise reduction, normalization, and pre-emphasis.

    Args:
        waveform: 1D numpy array, mono audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        dict with keys: waveform (processed), snr_estimated_db (float).
    """
    conf = pipeline_conf["pipeline"]

    # Denoise
    denoised = nr.reduce_noise(y=waveform, sr=sample_rate)

    # Estimate SNR
    noise_floor = np.mean(denoised[:sample_rate // 10] ** 2)
    signal_power = np.mean(denoised ** 2)
    noise_floor = max(noise_floor, 1e-10)
    snr_db = 10.0 * np.log10(signal_power / noise_floor)

    # Remove DC + normalize to unit variance
    denoised = denoised - denoised.mean()
    rms = np.sqrt(np.mean(denoised ** 2))
    if rms > 1e-10:
        denoised = denoised / rms

    # Pre-emphasis
    coef = conf["pre_emphasis_coef"]
    denoised = np.append(denoised[0], denoised[1:] - coef * denoised[:-1])

    return {
        "waveform": denoised,
        "snr_estimated_db": round(snr_db, 1),
    }
