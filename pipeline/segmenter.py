"""
Segment audio into fixed-length 3s non-overlapping frames.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
from utils.config_handler import pipeline_conf


def segment(waveform: np.ndarray, sample_rate: int) -> dict:
    """Zero-pad tail to nearest 3s multiple, then split via librosa.util.frame.

    Args:
        waveform: 1D numpy array of preprocessed audio.
        sample_rate: Sample rate in Hz.

    Returns:
        dict with keys: segments (list of 1D np arrays), num_segments (int),
        rms_per_segment (list of float), segments_dropped (int).
    """
    conf = pipeline_conf["pipeline"]
    seg_len = int(conf["segment_length_s"] * sample_rate)  # 48000 samples

    # Zero-pad tail to integer multiple of seg_len
    remainder = len(waveform) % seg_len
    if remainder != 0:
        pad_len = seg_len - remainder
        waveform = np.pad(waveform, (0, pad_len))

    # Non-overlapping frame split
    frames = librosa.util.frame(waveform, frame_length=seg_len, hop_length=seg_len, axis=0)
    segments = [frames[i] for i in range(frames.shape[0])]

    rms_per_segment = [round(float(np.sqrt(np.mean(s ** 2))), 4) for s in segments]
    segments_dropped = 0

    return {
        "segments": segments,
        "num_segments": len(segments),
        "rms_per_segment": rms_per_segment,
        "segments_dropped": segments_dropped,
    }
