# 水声目标识别 Agent MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 LangGraph Agent 骨架上，构建上传 wav → 预处理 → 多模态模型推理 → LLM 报告生成的完整闭环。

**Architecture:** 新增 `pipeline/` 模块（7 个纯 Python 文件）作为识别 Pipeline，封装为 `recognize_ship` LangChain tool 注册到现有 `ReactAgent`。LLM 负责意图理解和报告生成，Pipeline 负责确定性推理。新增 `rag_search` tool 复用现有 RAG 服务。

**Tech Stack:** librosa, noisereduce, torch, transformers (CLAP), 现有 LangGraph + Streamlit

---

### Task 1: 安装新依赖

**Files:** None

- [ ] **Step 1: 安装 librosa 和 noisereduce**

```bash
conda install -c conda-forge librosa
pip install noisereduce
```

- [ ] **Step 2: 验证 torch 和 transformers 可用**

```bash
python -c "import torch; print(torch.__version__); import transformers; print(transformers.__version__)"
```

如果报错，安装：
```bash
pip install torch torchaudio transformers
```

---

### Task 2: 创建 Pipeline 配置文件

**Files:**
- Create: `config/pipeline.yml`

- [ ] **Step 1: 创建 `config/pipeline.yml`**

```yaml
# Pipeline 配置文件 — 路径和模型参数

# 标注文件
annotation_csv: "X:/数据集/ShipEar/data_preprocessing/annotation/shipear_group_class_segmented_prompt_en_5_frame_Windows_16kHz_3s_0%.csv"

# 模型
model:
  source_path: "X:/Git_Clone/ShipsEar-Multimodal-Classification/src"
  checkpoint: "X:/Git_Clone/ShipsEar-Multimodal-Classification/saved/checkpoints/model_best.pt"
  clap_path: "X:/数据集/Research_Project/Zero-Shot_Project/clap-htsat-unfused"
  device: "cuda:0"
  fusion_embed_dim: 512
  fusion_heads: 8
  fusion_layers: 2
  num_classes: 5
  expert_num: 8
  top_k: 3

# 类别映射
class_names:
  0: "作业船"
  1: "小型船艇"
  2: "客船"
  3: "大型商船"
  4: "自然环境噪声"

# 预处理参数
preprocessing:
  sample_rate: 16000
  segment_duration_s: 3
  n_fft: 2048
  hop_length: 512
  n_mels: 128
  n_mfcc: 40
  mfcc_width: 94
  pre_emphasis_coef: 0.97
  noise_reduce:
    stationary: true
    time_mask_smooth_ms: 50
    freq_mask_smooth_hz: 500
    prop_decrease: 1.0

# 置信度阈值
low_confidence_threshold: 0.5
```

- [ ] **Step 2: 创建配置加载入口**

在 `utils/config_handler.py` 末尾追加：

```python
def load_pipeline_config() -> dict:
    """加载 Pipeline 配置文件"""
    import yaml
    from utils.path_tool import get_abs_path
    with open(get_abs_path("config/pipeline.yml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

pipeline_conf = load_pipeline_config()
```

- [ ] **Step 3: 验证配置加载成功**

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.').absolute())); from utils.config_handler import pipeline_conf; print(pipeline_conf['class_names'])"
```

- [ ] **Step 4: Commit**

```bash
git add config/pipeline.yml utils/config_handler.py
git commit -m "feat: add pipeline config and loader"
```

---

### Task 3: 音频加载模块

**Files:**
- Create: `pipeline/audio_loader.py`

- [ ] **Step 1: 创建 `pipeline/audio_loader.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np
from utils.config_handler import pipeline_conf


class AudioLoader:
    def __init__(self):
        self.target_sr = pipeline_conf["preprocessing"]["sample_rate"]

    def load(self, audio_path: str) -> dict:
        waveform, sr = librosa.load(
            audio_path, sr=self.target_sr, mono=True
        )
        duration_s = len(waveform) / self.target_sr

        if len(waveform) < self.target_sr * 1:  # < 1s
            raise ValueError(f"音频过短 ({duration_s:.1f}s)，至少需要 1 秒")

        return {
            "waveform": waveform,
            "sample_rate": self.target_sr,
            "duration_s": duration_s,
            "channels": 1,
            "filename": Path(audio_path).name,
        }
```

- [ ] **Step 2: 本地验证**

```bash
python -c "
import sys; from pathlib import Path; sys.path.insert(0, '.')
from pipeline.audio_loader import AudioLoader
loader = AudioLoader()
import librosa
import numpy as np
# 生成 5 秒 440Hz 测试音频
test_wav = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 80000))
import soundfile as sf
sf.write('test_tone.wav', test_wav, 16000)
info = loader.load('test_tone.wav')
print(f'sr={info[\"sample_rate\"]}, dur={info[\"duration_s\"]:.1f}s, ch={info[\"channels\"]}')
"
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/audio_loader.py
git commit -m "feat: add audio loader module"
```

---

### Task 4: 预处理模块（降噪 + 去均值归一化 + 预加重）

**Files:**
- Create: `pipeline/preprocessing.py`

- [ ] **Step 1: 创建 `pipeline/preprocessing.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import noisereduce as nr
from utils.config_handler import pipeline_conf


class Preprocessor:
    def __init__(self):
        cfg = pipeline_conf["preprocessing"]
        self.sr = cfg["sample_rate"]
        self.pre_emphasis_coef = cfg["pre_emphasis_coef"]
        self.nr_cfg = cfg["noise_reduce"]
        self.n_fft = cfg["n_fft"]
        self.hop_length = cfg["hop_length"]

    def process(self, waveform: np.ndarray) -> np.ndarray:
        # 1. 降噪
        denoised = nr.reduce_noise(
            y=waveform,
            sr=self.sr,
            stationary=self.nr_cfg["stationary"],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            time_mask_smooth_ms=self.nr_cfg["time_mask_smooth_ms"],
            freq_mask_smooth_hz=self.nr_cfg["freq_mask_smooth_hz"],
            prop_decrease=self.nr_cfg["prop_decrease"],
        )

        # 2. 去均值 + 单位方差归一化
        normalized = denoised - np.mean(denoised)
        std = np.std(normalized)
        if std > 1e-9:
            normalized = normalized / std

        # 3. 预加重 (y[t] = x[t] - coef * x[t-1])
        pre_emphasized = np.append(
            normalized[0],
            normalized[1:] - self.pre_emphasis_coef * normalized[:-1],
        )

        return pre_emphasized.astype(np.float32)
```

- [ ] **Step 2: 本地验证（使用上一步的 test_tone.wav）**

```bash
python -c "
import sys; from pathlib import Path; sys.path.insert(0, '.')
from pipeline.audio_loader import AudioLoader
from pipeline.preprocessing import Preprocessor
loader = AudioLoader()
info = loader.load('test_tone.wav')
proc = Preprocessor()
clean = proc.process(info['waveform'])
print(f'processed shape={clean.shape}, dtype={clean.dtype}, mean={clean.mean():.4f}')
assert clean.dtype == np.float32
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/preprocessing.py
git commit -m "feat: add preprocessing module (denoise + normalize + pre-emphasis)"
```

---

### Task 5: 分段模块

**Files:**
- Create: `pipeline/segmenter.py`

- [ ] **Step 1: 创建 `pipeline/segmenter.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
from utils.config_handler import pipeline_conf


class Segmenter:
    def __init__(self):
        cfg = pipeline_conf["preprocessing"]
        self.sr = cfg["sample_rate"]
        self.segment_len = cfg["segment_duration_s"] * self.sr  # 48000

    def segment(self, waveform: np.ndarray) -> list[np.ndarray]:
        # 尾部补零至 segment_len 整数倍
        remainder = len(waveform) % self.segment_len
        if remainder != 0:
            pad_len = self.segment_len - remainder
            waveform = np.pad(waveform, (0, int(pad_len)), mode="constant")

        # librosa.util.frame 切帧（非重叠）
        frames = librosa.util.frame(
            waveform, frame_length=int(self.segment_len), hop_length=int(self.segment_len)
        )
        # frames shape: (segment_len, n_segments)
        return [frames[:, i].copy() for i in range(frames.shape[1])]

    @property
    def segment_duration_s(self) -> float:
        return pipeline_conf["preprocessing"]["segment_duration_s"]
```

- [ ] **Step 2: 本地验证**

```bash
python -c "
import sys; from pathlib import Path; sys.path.insert(0, '.')
import numpy as np
from pipeline.segmenter import Segmenter
seg = Segmenter()
# 100000 samples = 2.083 segments at 48000 -> should pad to 3 segments
fake = np.random.randn(100000).astype(np.float32)
segments = seg.segment(fake)
print(f'segments={len(segments)}')
for i, s in enumerate(segments):
    print(f'  seg_{i}: len={len(s)}')
assert len(segments) == 3  # 100000 -> pad to 144000 -> 3 segments
assert all(len(s) == 48000 for s in segments)
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/segmenter.py
git commit -m "feat: add segmenter module (zero-pad + 3s fixed-length framing)"
```

---

### Task 6: 特征提取模块（Mel + MFCC）

**Files:**
- Create: `pipeline/feature_extractor.py`

- [ ] **Step 1: 创建 `pipeline/feature_extractor.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
from utils.config_handler import pipeline_conf


class FeatureExtractor:
    def __init__(self):
        cfg = pipeline_conf["preprocessing"]
        self.sr = cfg["sample_rate"]
        self.n_fft = cfg["n_fft"]
        self.hop_length = cfg["hop_length"]
        self.n_mels = cfg["n_mels"]
        self.n_mfcc = cfg["n_mfcc"]
        self.mfcc_width = cfg["mfcc_width"]

    def extract_mel(self, segment: np.ndarray) -> np.ndarray:
        """提取 Mel 谱: (n_mels, T)"""
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=False,
        )
        return librosa.power_to_db(mel, ref=np.max)

    def extract_mfcc(self, segment: np.ndarray) -> np.ndarray:
        """提取 MFCC: (n_mfcc, mfcc_width)"""
        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
        )
        return mfcc[:, :self.mfcc_width]

    def extract_all(self, segments: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        mels = []
        mfccs = []
        for seg in segments:
            mels.append(self.extract_mel(seg))
            mfccs.append(self.extract_mfcc(seg))
        return mels, mfccs
```

- [ ] **Step 2: 本地验证**

```bash
python -c "
import sys; from pathlib import Path; sys.path.insert(0, '.')
import numpy as np
from pipeline.feature_extractor import FeatureExtractor
ext = FeatureExtractor()
seg = np.random.randn(48000).astype(np.float32)
mel = ext.extract_mel(seg)
mfcc = ext.extract_mfcc(seg)
print(f'mel shape={mel.shape}')     # expect (128, T)
print(f'mfcc shape={mfcc.shape}')   # expect (40, 94)  
assert mel.shape[0] == 128
assert mfcc.shape == (40, 94)
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/feature_extractor.py
git commit -m "feat: add feature extractor (Mel 128 + MFCC 40x94)"
```

---

### Task 7: 标注匹配模块

**Files:**
- Create: `pipeline/metadata_lookup.py`

- [ ] **Step 1: 创建 `pipeline/metadata_lookup.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils.config_handler import pipeline_conf


class MetadataLookup:
    def __init__(self):
        csv_path = pipeline_conf["annotation_csv"]
        self.df = pd.read_csv(csv_path)
        self.df["_basename"] = self.df["Filename"].apply(lambda x: Path(str(x)).name)

    def lookup(self, filename: str) -> dict:
        basename = Path(filename).name
        match = self.df[self.df["_basename"] == basename]

        if match.empty:
            return {
                "channel_depth_m": None,
                "wind": None,
                "distance": None,
                "prompt_en": f"Underwater acoustic recording of an unknown marine vessel.",
                "source": "default",
            }

        row = match.iloc[0]
        return {
            "channel_depth_m": self._safe_float(row.get("Channel Depth")),
            "wind": self._safe_float(row.get("Wind")),
            "distance": self._safe_str(row.get("Distance")),
            "prompt_en": str(row["prompt_en"]),
            "source": "csv",
        }

    @staticmethod
    def _safe_float(val) -> float | None:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_str(val) -> str | None:
        if pd.isna(val):
            return None
        return str(val)
```

- [ ] **Step 2: 本地验证**

```bash
python -c "
import sys; from pathlib import Path; sys.path.insert(0, '.')
from pipeline.metadata_lookup import MetadataLookup
lookup = MetadataLookup()
# 测试 CSV 中存在的文件
result = lookup.lookup('10__10_07_13_marDeOnza_Sale.wav')
print(f'match: {result}')
assert result['source'] == 'csv'
assert result['prompt_en'].startswith('Hydrophone')
# 测试不存在的文件
result2 = lookup.lookup('nonexistent.wav')
print(f'no match: {result2}')
assert result2['source'] == 'default'
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/metadata_lookup.py
git commit -m "feat: add metadata lookup module (CSV annotation matching)"
```

---

### Task 8: 模型封装模块

**Files:**
- Create: `pipeline/model_wrapper.py`

- [ ] **Step 1: 创建 `pipeline/model_wrapper.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from utils.config_handler import pipeline_conf


class ShipClassifier:
    def __init__(self):
        cfg = pipeline_conf["model"]

        # 添加模型源码路径
        model_src = cfg["source_path"]
        if model_src not in sys.path:
            sys.path.insert(0, model_src)

        from model.Multimodal_MoE import MultimodalModel, ClapTextBranch

        self.device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

        self.model = MultimodalModel(
            fusion_embed_dim=cfg["fusion_embed_dim"],
            fusion_output_dim=cfg["fusion_embed_dim"],
            fusion_heads=cfg["fusion_heads"],
            fusion_layers=cfg["fusion_layers"],
            ff_multiplier=4,
            dropout=0.1,
            num_classes=cfg["num_classes"],
            time_branch=None,
            spectrogram_branch=None,
            text_branch=ClapTextBranch(
                pretrained_model_name=cfg["clap_path"],
                local_files_only=True,
                target_dim=cfg["fusion_embed_dim"],
            ),
            classifier_kwargs={
                "expert_num": cfg["expert_num"],
                "top_k": cfg["top_k"],
                "balance_wt": 0.01,
            },
        )

        checkpoint = torch.load(cfg["checkpoint"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model.to(self.device)
        self.model.eval()

        self.class_names = pipeline_conf["class_names"]

    def predict_batch(
        self,
        mels: list[np.ndarray],
        mfccs: list[np.ndarray],
        prompt: str,
    ) -> list[dict]:
        # 准备 tensor
        mel_tensors = []
        mfcc_tensors = []
        for mel, mfcc in zip(mels, mfccs):
            mel_t = torch.from_numpy(mel).float().unsqueeze(0)    # (1, 128, T)
            mfcc_t = torch.from_numpy(mfcc).float().unsqueeze(0)  # (1, 40, 94)
            mel_tensors.append(mel_t)
            mfcc_tensors.append(mfcc_t)

        mel_batch = torch.stack(mel_tensors).to(self.device)
        mfcc_batch = torch.stack(mfcc_tensors).to(self.device)

        with torch.no_grad():
            output = self.model(
                time=mel_batch,
                spectrogram=mfcc_batch,
                texts=prompt,
                return_attention=True,
            )

        logits = output["logits"].cpu()
        routing = output["routing_prob"].cpu()
        probs = torch.softmax(logits, dim=-1)

        # cross-attention 聚合
        cross_attn = output.get("cross_attention", [])
        attn_layers = []
        for layer_attn in cross_attn:
            layer_attn = layer_attn.cpu()  # (B, heads, 1, M)
            avg = layer_attn.mean(dim=(1, 2))  # (B, M)
            attn_layers.append(avg)

        # 获取各模态 embedding norm
        modality_norms = {}
        for key in ["time", "spectrogram", "text"]:
            if key in output:
                val = output[key]
                if isinstance(val, torch.Tensor):
                    modality_norms[key] = float(val.norm(dim=-1).mean().cpu().item())

        results = []
        for i in range(len(mels)):
            top5_idx = probs[i].argsort(descending=True)[:5].tolist()
            class_id = top5_idx[0]
            results.append({
                "class_id": class_id,
                "class_name": self.class_names.get(str(class_id), str(class_id)),
                "confidence": float(probs[i][class_id].item()),
                "top5_probs": {
                    self.class_names.get(str(j), str(j)): float(probs[i][j].item())
                    for j in top5_idx
                },
                "logits": logits[i].tolist(),
                "routing_prob": routing[i].tolist(),
                "cross_attention": [
                    {
                        "time": float(layer[i][0]) if layer.shape[1] > 0 else 0.0,
                        "spectrogram": float(layer[i][1]) if layer.shape[1] > 1 else 0.0,
                        "text": float(layer[i][2]) if layer.shape[1] > 2 else 0.0,
                    }
                    for layer in attn_layers
                ],
                "modality_norms": {
                    k: float(v) for k, v in modality_norms.items()
                },
                "fusion_norm": float(output["fusion"][i].norm().cpu().item()),
            })

        return results
```

- [ ] **Step 2: 验证模型加载**

```bash
python -c "
import sys; from pathlib import Path; sys.path.insert(0, '.')
from pipeline.model_wrapper import ShipClassifier
clf = ShipClassifier()
print(f'device={clf.device}')
print(f'classes={clf.class_names}')
print('Model loaded OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/model_wrapper.py
git commit -m "feat: add model wrapper (MultimodalModel load + batch inference)"
```

---

### Task 9: Pipeline 编排引擎

**Files:**
- Create: `pipeline/pipeline_engine.py`

- [ ] **Step 1: 创建 `pipeline/pipeline_engine.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from utils.config_handler import pipeline_conf
from utils.logger_handler import logger


class PipelineEngine:
    def __init__(self):
        # 延迟导入，只在首次 run() 时加载模型
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from pipeline.model_wrapper import ShipClassifier
            logger.info("[pipeline] 加载模型中...")
            self._model = ShipClassifier()
            logger.info("[pipeline] 模型加载完成")
        return self._model

    def run(self, audio_path: str) -> dict:
        t_start = time.time()

        try:
            # 1. 音频加载
            from pipeline.audio_loader import AudioLoader
            loader = AudioLoader()
            info = loader.load(audio_path)
            logger.info(f"[pipeline] ① 音频加载完成: {info['duration_s']:.1f}s, sr={info['sample_rate']}")

            # 2. 预处理
            from pipeline.preprocessing import Preprocessor
            preprocessor = Preprocessor()
            clean = preprocessor.process(info["waveform"])
            logger.info("[pipeline] ② 预处理完成")

            # 3. 分段
            from pipeline.segmenter import Segmenter
            segmenter = Segmenter()
            segments = segmenter.segment(clean)
            logger.info(f"[pipeline] ③ 分段完成: {len(segments)} 段")

            # 4. 特征提取
            from pipeline.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor()
            mels, mfccs = extractor.extract_all(segments)
            logger.info(f"[pipeline] ④ 特征提取完成: Mel {mels[0].shape}, MFCC {mfccs[0].shape}")

            # 5. 标注匹配
            from pipeline.metadata_lookup import MetadataLookup
            lookup = MetadataLookup()
            metadata = lookup.lookup(info["filename"])
            logger.info(f"[pipeline] ⑤ 标注匹配完成: source={metadata['source']}")

            # 6. 模型推理
            logger.info(f"[pipeline] ⑥ 开始推理 {len(segments)} 个 segment...")
            predictions = self.model.predict_batch(mels, mfccs, metadata["prompt_en"])
            logger.info(f"[pipeline] ⑥ 推理完成")

            # 7. 结果聚合
            aggregated = self._aggregate(predictions)
            logger.info(f"[pipeline] ⑦ 聚合完成: top={aggregated['top_class']} ({aggregated['top_confidence']:.2%})")

            # 计算预处理统计
            rms_per_segment = [float(np.sqrt(np.mean(s ** 2))) for s in segments]

            return {
                "status": "success",
                "audio_info": {
                    "filename": info["filename"],
                    "sample_rate": info["sample_rate"],
                    "duration_s": round(info["duration_s"], 1),
                    "channels": info["channels"],
                    "num_segments": len(segments),
                },
                "metadata": metadata,
                "predictions": predictions,
                "aggregated": aggregated,
                "internals": {
                    "cross_attention": {
                        f"layer_{i+1}": pred["cross_attention"][i]
                        for i, pred in enumerate([predictions[0]])
                        if pred.get("cross_attention")
                    },
                    "modality_embeddings_norm": predictions[0].get("modality_norms", {}),
                    "fusion_embedding_norm": predictions[0].get("fusion_norm", 0),
                    "preprocessing": {
                        "segments_dropped": 0,
                        "snr_estimated_db": None,
                        "rms_per_segment": rms_per_segment,
                    },
                },
                "inference_time_ms": int((time.time() - t_start) * 1000),
            }

        except Exception as e:
            logger.error(f"[pipeline] 失败: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "audio_info": None,
                "metadata": None,
                "predictions": [],
                "aggregated": None,
                "internals": None,
                "inference_time_ms": 0,
            }

    def _aggregate(self, predictions: list[dict]) -> dict:
        from collections import Counter
        threshold = pipeline_conf["low_confidence_threshold"]

        class_ids = [p["class_id"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
        vote = Counter(class_ids)
        class_name_vote = Counter(p["class_name"] for p in predictions)

        top_class_id = vote.most_common(1)[0][0]
        top_class_name = predictions[0]["top5_probs"].keys().__iter__().__next__()
        for p in predictions:
            if p["class_id"] == top_class_id:
                top_class_name = p["class_name"]
                break

        uncertain = [
            p["segment_id"]
            for p in predictions
            if p["confidence"] < threshold
        ]

        mean_conf_per_class = {}
        for cls_name in class_name_vote:
            cls_confs = [
                p["confidence"]
                for p in predictions
                if p["class_name"] == cls_name
            ]
            mean_conf_per_class[cls_name] = round(float(np.mean(cls_confs)), 4)

        return {
            "top_class": top_class_name,
            "top_confidence": round(float(np.mean([
                p["confidence"] for p in predictions if p["class_id"] == top_class_id
            ])), 4),
            "vote_distribution": dict(class_name_vote),
            "mean_confidence_per_class": mean_conf_per_class,
            "uncertain_segments": uncertain,
            "low_confidence_overall": all(c < threshold for c in confidences),
            "balance_loss": None,
        }
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/pipeline_engine.py
git commit -m "feat: add pipeline engine (orchestrate full recognition pipeline)"
```

---

### Task 10: recognize_ship LangChain Tool

**Files:**
- Create: `agent/tools/recognize_ship_tool.py`

- [ ] **Step 1: 创建 `agent/tools/recognize_ship_tool.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.tools import tool
from pipeline.pipeline_engine import PipelineEngine

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = PipelineEngine()
    return _engine

@tool
def recognize_ship(audio_path: str) -> dict:
    """对用户上传的水声录音进行目标识别。

    输入音频文件路径，自动完成预处理、特征提取、模型推理和结果聚合。
    返回结构化识别结果，包含：
    - audio_info: 音频基本信息（文件名、采样率、时长、段数）
    - metadata: 采集环境参数（水深、风速、距离、文本描述）
    - predictions: 每段的识别结果（类别、置信度、Top-5概率、模型内部信息）
    - aggregated: 聚合结果（投票分布、置信度均值、不确定段）
    - internals: 模型内部信息（跨模态注意力、各分支贡献度）

    Args:
        audio_path: 音频文件的绝对路径（.wav 格式，建议 16kHz 单通道）
    """
    engine = _get_engine()
    return engine.run(audio_path)
```

- [ ] **Step 2: Commit**

```bash
git add agent/tools/recognize_ship_tool.py
git commit -m "feat: add recognize_ship LangChain tool"
```

---

### Task 11: rag_search LangChain Tool

**Files:**
- Create: `agent/tools/rag_search_tool.py`

- [ ] **Step 1: 创建 `agent/tools/rag_search_tool.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService

_rag = None

def _get_rag():
    global _rag
    if _rag is None:
        _rag = RagSummarizeService()
    return _rag

@tool
def rag_search(query: str) -> str:
    """搜索水声领域知识库，获取专业参考资料。

    当需要解释模型的识别依据、对比不同船型的声学特征差异、
    或补充水声目标识别领域的背景知识时调用此工具。

    Args:
        query: 中文检索词，例如 "客船辐射噪声特征"、"LOFAR谱分析方法"
    Returns:
        基于知识库参考资料的摘要回答
    """
    rag = _get_rag()
    return rag.rag_summarize(query)
```

- [ ] **Step 2: Commit**

```bash
git add agent/tools/rag_search_tool.py
git commit -m "feat: add rag_search LangChain tool"
```

---

### Task 12: 更新 ReactAgent 注册工具

**Files:**
- Modify: `agent/react_agent.py`

- [ ] **Step 1: 修改 `agent/react_agent.py`**

将第 22 行 `tools=[]` 替换为：

```python
from agent.tools.recognize_ship_tool import recognize_ship
from agent.tools.rag_search_tool import rag_search

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[recognize_ship, rag_search],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
```

修改后的完整文件 `agent/react_agent.py`：

```python
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_tool import get_abs_path

import dotenv
dotenv.load_dotenv(get_abs_path("config/.env"))
os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("LLM_BASE_URL")

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools.recognize_ship_tool import recognize_ship
from agent.tools.rag_search_tool import rag_search

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[recognize_ship, rag_search],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, history: list = None):
        messages = list(history) if history else []
        messages.append({"role": "user", "content": query})
        input_dict = {
            "messages": messages
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"
```

- [ ] **Step 2: Commit**

```bash
git add agent/react_agent.py
git commit -m "feat: register recognize_ship and rag_search tools in ReactAgent"
```

---

### Task 13: 更新 Prompt 文件

**Files:**
- Modify: `prompts/main_prompt.txt`
- Modify: `prompts/report_prompt.txt`

- [ ] **Step 1: 重写 `prompts/main_prompt.txt`**

```
你是水声目标识别领域的专业智能助手。

你可以使用以下工具：

1. **recognize_ship** — 对水声录音进行自动目标识别。
   接收音频文件路径，自动完成预处理、特征提取、深度学习模型推理，
   返回结构化的识别结果（类别、置信度、分段统计、模型内部信息）。

2. **rag_search** — 搜索水声领域知识库。
   检索船舶声学特性、信号处理方法、目标识别原理等专业资料。

你的工作流程：
- 用户上传音频文件后，调用 recognize_ship 进行识别
- 根据识别结果生成专业的分析报告
- 如果用户追问技术细节，使用 rag_search 补充领域知识

专业领域包括：
- 水声学基础理论
- 水声信号处理（时频分析、DEMON谱、LOFAR谱）
- 声呐系统设计与应用
- 水下目标探测与识别（辐射噪声特征、船舶分类）
- 海洋声学环境
```

- [ ] **Step 2: 重写 `prompts/report_prompt.txt`**

```
你是专业的水声目标识别分析报告写手。根据识别结果撰写专业分析报告。

## 报告生成规则

当拿到 recognize_ship 返回的结构化结果后，按以下模板生成 Markdown 报告：

# 水声目标识别分析报告

## 一、音频概况
- 文件名、时长、采样率、通道数、切分段数
- 采集环境参数：水深/风速/距离（若标注文件中无匹配则为"未知"）

## 二、识别结论
- **最终判定**：[类别名称] | 综合置信度：[XX.X%]
- **置信度评级**：≥85% 高 / 70-85% 中 / <70% 低
- 投票分布表（各段判定统计）

## 三、分段识别详情
- 表格列出所有段的：段号 | 判定类别 | 置信度
- 标注低置信度段（<50%）

## 四、模型决策依据
- 基于 cross_attention 权重分析 Mel/MFCC/Text 各模态的贡献
- 主导模态及解读

## 五、可信度评估
- 综合置信度分析
- 若全段置信度均低，明确说明并建议人工复核

## 六、专业建议
- 基于识别结果给出可操作建议

## 写作原则
- 使用水声领域标准术语
- 置信度保留一位小数
- 对不确定结果诚实标注，不掩饰
- 百分比、表格、数字用 Markdown 格式化
```

- [ ] **Step 3: Commit**

```bash
git add prompts/main_prompt.txt prompts/report_prompt.txt
git commit -m "feat: update prompts for recognize_ship and rag_search tools"
```

---

### Task 14: 更新 Streamlit UI（文件上传 + 报告渲染）

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 重写 `app.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import tempfile
import streamlit as st
from agent.react_agent import ReactAgent
from rag.file_chat_history_store import FileChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

HISTORY_FILE = FileChatMessageHistory("001.json", "rag/chat_history")

st.set_page_config(page_title="水声目标识别智能Agent", layout="wide")
st.title("水声目标识别智能Agent")

# ---- 侧边栏：文件上传 ----
with st.sidebar:
    st.header("音频上传")
    uploaded_file = st.file_uploader("上传 .wav 文件", type=["wav"])

    if uploaded_file is not None:
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success(f"已上传: {uploaded_file.name}")

        if st.button("开始识别", type="primary", use_container_width=True):
            with st.spinner("识别中..."):
                agent = st.session_state.get("agent", ReactAgent())
                query = f"请识别这个音频文件：{tmp_path}"
                # 直接获取完整响应
                result_parts = []
                for chunk in agent.execute_stream(query, history=[]):
                    result_parts.append(chunk)
                full_response = "".join(result_parts)

            st.session_state["last_report"] = full_response
            st.session_state["last_audio_name"] = uploaded_file.name
            st.rerun()

    st.divider()

    if st.button("清空对话历史"):
        HISTORY_FILE.clear()
        st.session_state["messages"] = []
        st.session_state["last_report"] = None
        st.success("历史记录已清空！")
        st.rerun()

# ---- 主区域：对话 + 报告 ----
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    loaded = []
    for msg in HISTORY_FILE.messages:
        if isinstance(msg, HumanMessage):
            loaded.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            loaded.append({"role": "assistant", "content": msg.content})
    st.session_state["messages"] = loaded

if "last_report" not in st.session_state:
    st.session_state["last_report"] = None

# 渲染已有对话
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 渲染上一次识别报告（若有）
if st.session_state.get("last_report"):
    with st.container(border=True):
        st.markdown(st.session_state["last_report"])

# 用户文本输入
prompt = st.chat_input("输入问题，或通过左侧上传音频后点击「开始识别」")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.spinner("Agent 思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            prompt,
            history=st.session_state["messages"][:-1],
        )

        response_parts = []
        def capture(generator, cache):
            for chunk in generator:
                if chunk.strip() == prompt:
                    continue
                if "Question:" not in chunk:
                    cache.append(chunk)
                    for char in chunk:
                        time.sleep(0.01)
                        yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_parts))
        full_response = "".join(response_parts)
        st.session_state["messages"].append({"role": "assistant", "content": full_response})

        HISTORY_FILE.add_messages([
            HumanMessage(content=prompt),
            AIMessage(content=full_response),
        ])
        st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add file upload and report rendering to Streamlit UI"
```

---

### Task 15: 端到端验证

**Files:** None (manual testing)

- [ ] **Step 1: 启动 Streamlit 应用**

```bash
streamlit run app.py
```

- [ ] **Step 2: 验证功能清单**

按以下 checklist 逐项验证：

1. 页面正常加载，左侧边栏有文件上传组件
2. 上传一个 16kHz 单通道 .wav 文件（可用 ShipsEar 数据集中的文件）
3. 点击"开始识别"，控制台输出 Pipeline 各步骤日志
4. 主区域渲染出识别报告，包含：
   - 音频概况
   - 识别结论（类别 + 置信度）
   - 分段详情表格
   - 模型决策依据
5. 在聊天框追问"为什么判为这个类别？"，Agent 基于已有结果或 RAG 回答
6. 测试错误情况：上传一个非 16kHz 文件，Agent 应给出友好提示
7. 测试清空历史按钮

- [ ] **Step 3: 记录问题并修复**

验证过程中若发现问题，记录并修复后重新验证。

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "chore: end-to-end verification fixes"
```

---

### Task 16: 更新 CLAUDE.md（记录新增模块）

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 在 CLAUDE.md 的 Architecture 一节末尾追加**

```
  pipeline/                            [新增] 水声目标识别 Pipeline
       audio_loader.py                 librosa 音频加载（16kHz 单通道）
       preprocessing.py                noisereduce 降噪 + 去均值归一化 + 预加重(0.97)
       segmenter.py                    尾部补零 + 3s 固定段长切分(librosa.util.frame)
       feature_extractor.py            Mel(128 bins) + MFCC(40×94, center=False)
       metadata_lookup.py              按文件名匹配标注 CSV，获取 prompt_en + 环境参数
       model_wrapper.py                MultimodalModel 加载 + predict_batch() 推理
       pipeline_engine.py              run(audio_path) → dict 编排入口
  agent/tools/
       recognize_ship_tool.py          [新增] recognize_ship LangChain tool
       rag_search_tool.py             [新增] RAG 检索 tool
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with new pipeline and tool modules"
```
