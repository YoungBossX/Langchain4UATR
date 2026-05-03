"""
Model loading and inference wrapper for MultimodalModel.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from utils.config_handler import pipeline_conf


_model = None
_class_names = None


def _get_model():
    """Lazy-load the MultimodalModel from checkpoint (cached)."""
    global _model
    if _model is not None:
        return _model

    conf = pipeline_conf["model"]
    model_src = Path(conf["model_src_path"])
    if str(model_src) not in sys.path:
        sys.path.insert(0, str(model_src))

    from model.Multimodal_MoE import MultimodalModel

    model = MultimodalModel(
        fusion_embed_dim=conf["fusion_embed_dim"],
        fusion_heads=conf["fusion_heads"],
        fusion_layers=conf["fusion_layers"],
        num_classes=conf["num_classes"],
        classifier_kwargs={
            "input_dim": conf["fusion_embed_dim"],
            "output_dim": conf["num_classes"],
            "expert_num": conf["expert_num"],
            "top_k": conf["top_k"],
        },
    )

    checkpoint_path = conf["checkpoint_path"]
    ckpt = torch.load(checkpoint_path, map_location=conf["device"], weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    model.to(conf["device"])
    model.eval()
    _model = model
    return _model


def get_class_names() -> dict:
    """Return class_id -> class_name mapping."""
    global _class_names
    if _class_names is None:
        _class_names = {int(k): v for k, v in pipeline_conf["class_names"].items()}
    return _class_names


def infer_segment(mel: np.ndarray, mfcc: np.ndarray, prompt_en: str) -> dict:
    """Run inference on a single 3s segment.

    Args:
        mel: np.ndarray shape (1, 128, T_mel).
        mfcc: np.ndarray shape (1, 1, 40, 94).
        prompt_en: Text description string for the CLAP text branch.

    Returns:
        dict with keys: class_id, class_name, confidence, top5_probs,
        logits, routing_prob, cross_attention, modality_embeddings_norm,
        fusion_embedding_norm.
    """
    conf = pipeline_conf["model"]
    device = conf["device"]
    model = _get_model()
    class_names = get_class_names()

    mel_t = torch.from_numpy(mel).to(device)
    mfcc_t = torch.from_numpy(mfcc).to(device)

    with torch.no_grad():
        outputs = model(
            time=mel_t,
            spectrogram=mfcc_t,
            texts=prompt_en,
            return_attention=True,
        )

    logits = outputs["logits"].squeeze(0).cpu().numpy()
    probs = torch.softmax(outputs["logits"].squeeze(0), dim=-1).cpu().numpy()
    top_idx = int(np.argmax(probs))

    # Top-5 probabilities
    sorted_idx = np.argsort(probs)[::-1]
    top5 = {class_names[int(i)]: round(float(probs[i]), 4) for i in sorted_idx[:5]}

    # Routing probability from MoE
    routing = outputs.get("routing_prob")
    routing_list = routing.squeeze(0).cpu().numpy().tolist() if routing is not None else []

    # Cross-attention weights
    cross_attn = {}
    if "cross_attention" in outputs:
        modality_names = outputs.get("modality_names", ["Time", "Spectrogram", "Text"])
        for layer_idx, attn in enumerate(outputs["cross_attention"]):
            # attn: (B, heads, 1, num_modalities) → average over heads
            avg = attn.squeeze(2).mean(dim=1).squeeze(0).cpu().numpy()
            cross_attn[f"layer_{layer_idx + 1}"] = {
                modality_names[i].lower(): round(float(avg[i]), 4)
                for i in range(len(modality_names))
            }

    # Modality embedding norms
    modality_norms = {}
    for key in ["time", "spectrogram", "text"]:
        if key in outputs:
            modality_norms[key] = round(float(torch.norm(outputs[key]).item()), 1)

    fusion_norm = round(float(torch.norm(outputs["fusion"]).item()), 1)

    return {
        "class_id": top_idx,
        "class_name": class_names[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "top5_probs": top5,
        "logits": [round(float(x), 4) for x in logits.tolist()],
        "routing_prob": [round(float(x), 4) for x in routing_list],
        "cross_attention": cross_attn,
        "modality_embeddings_norm": modality_norms,
        "fusion_embedding_norm": fusion_norm,
    }
