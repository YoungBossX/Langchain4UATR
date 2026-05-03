"""
Pipeline orchestration engine: load → preprocess → segment → extract → infer → aggregate.
Runs in pytorch_env via subprocess. CLI entry for cross-environment invocation.
"""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline.audio_loader import load_audio
from pipeline.preprocessing import preprocess
from pipeline.segmenter import segment
from pipeline.feature_extractor import extract_features
from pipeline.metadata_lookup import lookup_metadata
from pipeline.model_wrapper import infer_segment, get_class_names
from utils.config_handler import pipeline_conf


class PipelineEngine:
    """Orchestrate the full underwater acoustic recognition pipeline."""

    def run(self, audio_path: str) -> dict:
        t_start = time.perf_counter()

        # Stage 1: Load audio
        audio = load_audio(audio_path)
        if audio.get("status") == "error":
            return {**audio, "inference_time_ms": 0}
        waveform = audio.pop("waveform")
        sample_rate = audio["sample_rate"]
        audio["num_segments"] = 0  # placeholder, updated after segmentation

        # Stage 2: Preprocess
        pre = preprocess(waveform, sample_rate)

        # Stage 3: Segment
        seg = segment(pre["waveform"], sample_rate)

        audio["num_segments"] = seg["num_segments"]

        # Stage 4: Metadata lookup
        metadata = lookup_metadata(audio["filename"])

        # Stage 5 & 6: Feature extraction + Inference per segment
        predictions = []
        for i, seg_waveform in enumerate(seg["segments"]):
            feats = extract_features(seg_waveform, sample_rate)
            pred = infer_segment(feats["mel"], feats["mfcc"], metadata["prompt_en"])
            pred["segment_id"] = i + 1
            predictions.append(pred)

        # Stage 7: Aggregate
        aggregated = _aggregate(predictions, seg)

        t_ms = round((time.perf_counter() - t_start) * 1000)

        return {
            "status": "success",
            "error_message": None,
            "audio_info": audio,
            "metadata": metadata,
            "predictions": predictions,
            "aggregated": aggregated,
            "internals": _build_internals(predictions, pre, seg),
            "inference_time_ms": t_ms,
        }


def _aggregate(predictions: list, seg_info: dict) -> dict:
    """Aggregate per-segment predictions into overall results."""
    class_names = get_class_names()
    num_classes = len(class_names)

    # Vote distribution
    vote_dist = {}
    for p in predictions:
        name = p["class_name"]
        vote_dist[name] = vote_dist.get(name, 0) + 1

    # Mean confidence per class
    conf_sum = {name: 0.0 for name in class_names.values()}
    conf_count = {name: 0 for name in class_names.values()}
    for p in predictions:
        name = p["class_name"]
        conf_sum[name] += p["confidence"]
        conf_count[name] += 1
    mean_conf = {
        name: round(conf_sum[name] / conf_count[name], 4) if conf_count[name] > 0 else 0.0
        for name in class_names.values()
    }

    # Top class by vote
    top_class = max(vote_dist, key=vote_dist.get)
    top_conf = mean_conf[top_class]

    # Uncertain segments (confidence < 0.5)
    uncertain = [p["segment_id"] for p in predictions if p["confidence"] < 0.5]

    # Low confidence overall (mean top-class confidence < 0.6)
    low_conf = top_conf < 0.6

    # Balance loss from last segment's MoE
    balance_loss = 0.0

    return {
        "top_class": top_class,
        "top_confidence": round(top_conf, 4),
        "vote_distribution": vote_dist,
        "mean_confidence_per_class": mean_conf,
        "uncertain_segments": uncertain,
        "low_confidence_overall": low_conf,
        "balance_loss": balance_loss,
    }


def _build_internals(predictions: list, pre_info: dict, seg_info: dict) -> dict:
    """Build the internals section for debugging and explanation."""
    # Collect cross-attention from first segment
    ca = predictions[0].get("cross_attention", {}) if predictions else {}

    # Collect modality norms from first segment
    mn = predictions[0].get("modality_embeddings_norm", {}) if predictions else {}
    fn = predictions[0].get("fusion_embedding_norm", 0.0) if predictions else 0.0

    return {
        "cross_attention": ca,
        "modality_embeddings_norm": mn,
        "fusion_embedding_norm": fn,
        "preprocessing": {
            "segments_dropped": seg_info["segments_dropped"],
            "snr_estimated_db": pre_info["snr_estimated_db"],
            "rms_per_segment": seg_info["rms_per_segment"],
        },
    }


if __name__ == "__main__":
    import json as _json
    engine = PipelineEngine()
    result = engine.run(sys.argv[1])
    print(_json.dumps(result, ensure_ascii=False, indent=2))
