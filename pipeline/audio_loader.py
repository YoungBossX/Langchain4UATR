"""
Audio file loading with librosa.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import soundfile as sf
from utils.config_handler import pipeline_conf


def load_audio(audio_path: str) -> dict:
    """Load and validate a wav file for the recognition pipeline.

    Returns:
        dict with keys: waveform (np.ndarray), sample_rate (int),
        duration_s (float), channels (int), filename (str)
        On error: status="error", error_message=str
    """
    conf = pipeline_conf["pipeline"]
    target_sr = conf["sample_rate"]

    try:
        info = sf.info(audio_path)
    except Exception as e:
        return {"status": "error", "error_message": f"无法读取音频文件: {e}"}

    try:
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=conf["mono"])
    except Exception as e:
        return {"status": "error", "error_message": f"音频解码失败: {e}"}

    if waveform.size == 0:
        return {"status": "error", "error_message": "音频文件为空"}

    duration = len(waveform) / sr

    return {
        "waveform": waveform,
        "sample_rate": sr,
        "duration_s": round(duration, 2),
        "channels": info.channels,
        "filename": Path(audio_path).name,
    }
