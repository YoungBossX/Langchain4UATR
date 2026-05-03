# 水声目标识别智能 Agent (Langchain4UATR)

基于 LangGraph + Streamlit 的水声目标识别智能助手。上传 wav 录音 → 自动预处理 → 多模态模型推理 → LLM 生成识别报告。

## 快速开始

### 环境准备

项目使用两个独立 conda 环境：

```bash
# Agent 环境（Streamlit + LangGraph）
conda create -n agent python=3.11
conda activate agent
pip install streamlit python-dotenv pyyaml
pip install langchain langchain-core langchain-openai langchain-community langchain-chroma langchain-text-splitters langgraph

# PyTorch 环境（模型推理 Pipeline）
conda create -n pytorch_env python=3.10
conda activate pytorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers librosa noisereduce soundfile scikit-learn pandas
```

### 配置

创建 `config/.env`：

```
LLM_API_KEY=<your-api-key>
LLM_BASE_URL=<your-api-base-url>
```

### 启动

```bash
conda run -n agent streamlit run app.py
```

在侧边栏上传 wav 文件，对话中说"识别这个文件"即可开始分析。

## 架构

```
Streamlit UI (app.py)
  ├─ 侧边栏：wav 文件上传
  └─ 主区域：对话 + 报告渲染
       └─ ReactAgent (agent/react_agent.py)
            ├─ recognize_ship  tool ──→ subprocess → conda run pytorch_env → pipeline/pipeline_engine.py
            └─ rag_search     tool ──→ rag/rag_service.py → ChromaDB

Pipeline (pipeline/, 运行于 pytorch_env)
  audio_loader → preprocessing → segmenter → feature_extractor → metadata_lookup → model_wrapper → aggregate
```

## Pipeline 预处理参数

| 参数 | 值 |
|------|-----|
| 采样率 | 16000 Hz |
| 通道 | 单通道 |
| 段长 | 3s (48000 samples) |
| Mel bins | 128 |
| MFCC | 40 维 × 94 帧 |
| 降噪 | noisereduce.reduce_noise() |
| 切段策略 | 补零至 3s 整数倍 → librosa.util.frame |

## 五分类系统

| class_id | 类别 | 包含船型 |
|----------|------|----------|
| 0 | Working vessels | Tugboat, Trawler, Mussel boat, Fishboat, Dredger |
| 1 | Small vessels | Motorboat, Pilot ship, Sailboat |
| 2 | Passenger vessels | Passengers |
| 3 | Large commercial | Ocean liner, RORO |
| 4 | Natural ambient noise | Natural ambient noise |

## 模型

- 架构：TimeConformerBranch + SpectrogramCNNBranch + ClapTextBranch → CrossAttentionFusionHead → TopKMoEClassifier
- Checkpoint：`X:\Git_Clone\ShipsEar-Multimodal-Classification\saved\checkpoints\model_best.pt`
- CLAP：`X:/数据集/Research_Project/Zero-Shot_Project/clap-htsat-unfused`
- 标注：`X:/数据集/ShipEar/data_preprocessing/annotation/shipear_group_class_segmented_prompt_en_5_frame_Windows_16kHz_3s_0%.csv`

## 依赖外部路径

| 资源 | 路径 |
|------|------|
| 模型 checkpoint | `X:\Git_Clone\ShipsEar-Multimodal-Classification\...\model_best.pt` |
| 模型源码 | `X:\Git_Clone\ShipsEar-Multimodal-Classification\src\model\` |
| CLAP 预训练模型 | `X:\数据集\Research_Project\Zero-Shot_Project\clap-htsat-unfused` |
| 标注 CSV | `X:\数据集\ShipEar\data_preprocessing\annotation\...csv` |
