# 水声目标识别 Agent MVP 设计文档

> **Status: ✅ 已实现** — 2026-05-03，commit `ffeee89`。实现与设计有以下偏差：
> - `predictions` 仅返回置信度最高的一段（非全部段），聚合仍用全段
> - `internals` 新增 `segment_count`、`top_segment_id`、`top_segment_confidence`
> - `app.py` 改为后台线程执行 + 轮询模式（支持停止按钮），停止按钮在输入框上方
> - `ChatOpenAI` 开启 `streaming=True`

## 概述

构建"任务理解—工具调用—模型推理—结果解释—报告生成"的智能闭环系统。MVP 采用 Hybrid 架构：确定性推理走 Pipeline，LLM 负责意图理解 + 结果解释 + 报告生成 + 追问。

## 架构

```
Streamlit UI (app.py)
  └─ ReactAgent (agent/react_agent.py)           LangGraph agent + SiliconFlow LLM
       ├─ recognize_ship 工具                    调用完整识别 Pipeline
       └─ rag_search 工具                        复用已有 RAG 服务

Pipeline Engine (pipeline/pipeline_engine.py)      纯 Python 模块，不依赖 LangChain
  ├─ audio_loader.py                             librosa 加载 wav
  ├─ preprocessing.py                            noisereduce 降噪 + 归一化 + 预加重
  ├─ segmenter.py                                补零至 3s 整数倍 + 切段
  ├─ feature_extractor.py                        Mel(128 bins) + MFCC(40×94) 提取
  ├─ metadata_lookup.py                          按文件名查 CSV 获取 prompt_en + 环境参数
  ├─ model_wrapper.py                            MultimodalModel 加载 + 推理
  └─ pipeline_engine.py                          编排入口 run()

Model: MultimodalModel (已有)
  ├─ TimeConformerBranch (Mel) → 512-dim
  ├─ SpectrogramCNNBranch (MFCC) → 512-dim
  ├─ ClapTextBranch (prompt_en) → 512-dim
  ├─ CrossAttentionFusionHead (2 layers, 8 heads)
  └─ TopKMoEClassifier (8 experts, top_k=3) → 5 classes
```

## Pipeline 输入输出 Schema

### 输入

```python
{
    "audio_path": "path/to/uploaded.wav",
    "annotation_csv": "path/to/shipear_annotations.csv",
}
```

### 输出

```python
{
    "status": "success" | "error",
    "error_message": null | "具体错误信息",
    "audio_info": {
        "filename": "...",
        "sample_rate": 16000,
        "duration_s": 45.2,
        "channels": 1,
        "num_segments": 15
    },
    "metadata": {
        "channel_depth_m": 4.8,       # CSV 匹配 → float | None
        "wind": 0,
        "distance": "<50",
        "prompt_en": "Hydrophone recording of...",
        "source": "csv" | "default"   # CSV 匹配到还是用的默认值
    },
    "predictions": [{
        "segment_id": 1,
        "class_id": 2,
        "class_name": "Passenger",
        "confidence": 0.87,
        "top5_probs": {"Passenger": 0.87, ...},
        "logits": [0.12, 0.45, 2.31, -0.15, -1.02],
        "routing_prob": [0.02, 0.01, 0.35, ...]
    }],
    "aggregated": {
        "top_class": "Passenger",
        "top_confidence": 0.82,
        "vote_distribution": {"Passenger": 12, "Motorboat": 2, ...},
        "mean_confidence_per_class": {"Passenger": 0.79, ...},
        "uncertain_segments": [3, 7],
        "low_confidence_overall": false,
        "balance_loss": 0.023
    },
    "internals": {
        "cross_attention": {
            "layer_1": {"time": 0.42, "spectrogram": 0.35, "text": 0.23},
            "layer_2": {"time": 0.38, "spectrogram": 0.33, "text": 0.29}
        },
        "modality_embeddings_norm": {"time": 12.4, "spectrogram": 8.7, "text": 5.2},
        "fusion_embedding_norm": 15.1,
        "preprocessing": {
            "segments_dropped": 0,
            "snr_estimated_db": 12.3,
            "rms_per_segment": [0.15, ...]
        }
    },
    "inference_time_ms": 234
}
```

`internals` 内所有值均为 JSON 可序列化类型（float/str/list），不做 tensor 直传。

## 预处理参数（固定）

| 参数 | 值 |
|---|---|
| 采样率 | 16000 Hz |
| 通道 | 单通道 |
| 段长 | 3s (48000 samples) |
| Mel bins | 128 |
| MFCC 维度 | 40 维 × 94 帧 |
| 降噪 | `noisereduce.reduce_noise()` |
| 归一化 | 去均值 + 单位方差 |
| 预加重 | 系数 0.97 |
| 切段策略 | 尾部补零至 3s 整数倍，再用 `librosa.util.frame` 切帧 |

## Tool 层

### recognize_ship

```python
@tool
def recognize_ship(audio_path: str) -> dict:
    """对上传的水声录音进行目标识别，返回结构化结果 dict"""
```

### rag_search

```python
@tool
def rag_search(query: str) -> str:
    """搜索水声领域知识库，返回基于参考资料的摘要回答"""
```

Agent 注册：`tools=[recognize_ship, rag_search]`，替换当前 `tools=[]`。

## Agent 决策流程

1. 用户上传 wav / 输入"识别这个文件"
2. LLM → 调用 `recognize_ship(audio_path)`
3. 拿到 dict → LLM 解释结果 → 流式生成 markdown 报告
4. 用户追问 → LLM 基于已有 `internals` 回答；必要时调 `rag_search`
5. 新一轮识别

## 报告模板

LLM 按以下章节生成 markdown：
1. 音频概况
2. 识别结论
3. 分段分析（表格）
4. 不确定段分析
5. 模型决策依据（基于 cross-attention）
6. 专业建议

## 错误处理

| 阶段 | 问题 | 策略 |
|---|---|---|
| 音频加载 | 格式不支持/损坏/采样率不符 | `status: "error"` + 原因，LLM 告知用户 |
| 预处理 | 音频 <3s / 全静音 | <3s 不切段处理；静音正常推理 |
| 标注匹配 | CSV 无匹配 | metadata 填 null + `source: "default"` |
| 模型推理 | GPU OOM / .pt 缺失 | 捕获异常 → error，建议切换 CPU |
| 结果聚合 | 全段低置信度 | 设 `low_confidence_overall: true` |

## 模型依赖

- Checkpoint: `X:\Git_Clone\ShipsEar-Multimodal-Classification\saved\checkpoints\model_best.pt`
- CLAP: `X:/数据集/Research_Project/Zero-Shot_Project/clap-htsat-unfused`
- 分类名映射: `["A", "B", "C", "D", "E"]` → 需替换为真实类别名
- 推理设备: GPU (cuda:0)

## MVP 范围边界

**做：**
- 单文件 wav 上传识别
- 3s 固定段长
- CSV 匹配已有标注
- 流式进度提示
- Markdown 报告 + 分段表格
- GPU 推理
- 对话追问（单轮，基于已有结果）
- RAG 检索

**不做（延后，接口预留）：**
- 文件夹批量识别
- 自适应段长
- 文本描述实时生成
- 可视化图表
- CPU 回退
- 多轮复杂对话
- 新领域文献入库

## 新增文件清单

```
pipeline/
    audio_loader.py
    preprocessing.py
    segmenter.py
    feature_extractor.py
    metadata_lookup.py
    model_wrapper.py
    pipeline_engine.py
agent/tools/
    recognize_ship_tool.py     # 新增
    rag_search_tool.py         # 新增
config/
    pipeline.yml               # 新增：pipeline 配置路径
```

## 修改文件清单

```
app.py                         # UI：文件上传 + 报告渲染区
agent/react_agent.py           # tools=[recognize_ship, rag_search]
prompts/main_prompt.txt        # 适配新工具描述
prompts/report_prompt.txt      # 适配真实 pipeline 输出
```

## 5 分类系统

| class_id | 类别名 | 包含船型 |
|---|---|---|
| 0 | Working vessels | Tugboat, Trawler, Mussel boat, Fishboat, Dredger |
| 1 | Small vessels | Motorboat, Pilot ship, Sailboat |
| 2 | Passenger vessels | Passengers |
| 3 | Large commercial | Ocean liner, RORO |
| 4 | Natural ambient noise | Natural ambient noise |

## 技术约束

- 不使用 `__init__.py`，沿用 `sys.path.insert` 模式
- 依赖通过 conda + pip 手动安装，无 requirements.txt
- LLM: SiliconFlow API (OpenAI 兼容)，模型 `Qwen/Qwen3-8B`
- 新增依赖: `librosa`, `noisereduce`, `soundfile`, `torch`, `transformers`
