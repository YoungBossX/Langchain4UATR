# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 启动命令

```bash
streamlit run app.py
```

没有 build step、linter、测试套件。依赖通过 conda 管理，两个独立环境：

```bash
# Agent 环境（Streamlit + LangGraph）
conda install streamlit python-dotenv pyyaml
pip install langchain langchain-core langchain-openai langchain-community langchain-chroma langchain-text-splitters langgraph

# PyTorch 环境（模型推理）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers librosa noisereduce soundfile scikit-learn pandas
```

## 环境配置

创建 `config/.env`：

```
LLM_API_KEY=<your-api-key>
LLM_BASE_URL=<your-api-base-url>
```

启动时 `agent/react_agent.py:9-10` 将其映射为 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`。默认聊天模型为 `Qwen/Qwen3-8B`，通过 OpenAI 兼容代理访问（在 `config/rag.yml` 中配置）。

## 架构

```
app.py (Streamlit UI，含侧边栏 wav 上传)
  └─ agent/react_agent.py           ReactAgent：LangGraph create_agent()
       ├─ agent/tools/recognize_ship_tool.py  @tool：subprocess → conda run pytorch_env → pipeline
       ├─ agent/tools/rag_search_tool.py      @tool：调用已有 RAG 服务
       ├─ agent/tools/middleware.py   三个 middleware：工具日志、模型调用日志、动态 prompt 切换
       ├─ model/factory.py            ChatOpenAI / OpenAIEmbeddings 工厂（单例）
       └─ prompts/*.txt              系统提示词（由 utils/prompt_loader.py 加载）

  pipeline/                          识别 Pipeline（在 pytorch_env 中运行）
       ├─ pipeline_engine.py          编排入口 + CLI（`python pipeline_engine.py <audio_path>` → JSON stdout）
       ├─ audio_loader.py             librosa 加载 wav
       ├─ preprocessing.py            noisereduce 降噪 + 归一化 + 预加重
       ├─ segmenter.py                补零至 3s 整数倍 + librosa.util.frame 切段
       ├─ feature_extractor.py        Mel(128 bins) + MFCC(40×94)，center=False
       ├─ metadata_lookup.py          按文件名查 CSV 获取 prompt_en + 环境参数
       └─ model_wrapper.py            加载 MultimodalModel + 推理

  rag/rag_service.py                 RAG 链：retrieve → prompt → LLM → StrOutputParser
       └─ rag/vector_store.py         ChromaDB 封装：MD5 去重、RecursiveCharacterTextSplitter、分批插入

  rag/file_chat_history_store.py      FileChatMessageHistory：JSON 文件对话持久化

  utils/
       config_handler.py              YAML 配置加载器 → 模块级 dict
       prompt_loader.py               读取 .txt prompt 文件
       path_tool.py                   项目根目录解析 + 绝对路径转换
       file_handler.py                MD5 哈希、PDF/TXT 文档加载
       logger_handler.py              双 Handler 日志器
```

### 跨环境通信

Agent 环境通过 subprocess 调用 pytorch_env 中的 Pipeline：

```python
# agent/tools/recognize_ship_tool.py
subprocess.run(
    ["conda", "run", "-n", "pytorch_env", "python", "pipeline/pipeline_engine.py", audio_path],
    capture_output=True, text=True, timeout=300
)
```

Pipeline 以 JSON 格式输出结果到 stdout，Agent 解析后返回结构化 dict。

### 模型信息

- **架构**: TimeConformerBranch (Mel) + SpectrogramCNNBranch (MFCC) + ClapTextBranch → CrossAttentionFusionHead (2层8头) → TopKMoEClassifier (8 experts, top_k=3)
- **输入**: Mel (1,128,T) + MFCC (1,1,40,94) + 文本 prompt
- **输出**: 5分类 — Working vessels (0), Small vessels (1), Passenger vessels (2), Large commercial (3), Natural ambient noise (4)
- **Checkpoint**: `X:\Git_Clone\ShipsEar-Multimodal-Classification\saved\checkpoints\model_best.pt`
- **CLAP**: `X:/数据集/Research_Project/Zero-Shot_Project/clap-htsat-unfused`
- **预处理**: 16kHz, mono, 3s段, Mel 128 bins, MFCC 40×94, center=False

### 提示词文件

| 文件 | 用途 |
|---|---|
| `prompts/main_prompt.txt` | 默认系统提示词：水声领域专家 agent，含 recognize_ship 和 rag_search 工具描述 |
| `prompts/rag_summarize.txt` | RAG 总结提示词：仅中文、基于事实、纯文本输出 |
| `prompts/report_prompt.txt` | 报告生成提示词，含报告模板（6章）。当 `runtime.context["report"]=True` 时由 middleware 动态加载 |

### 配置体系

| 文件 | 关键配置 |
|---|---|
| `config/rag.yml` | `chat_model_name: Qwen/Qwen3-8B`、`embedding_model_name` |
| `config/chroma.yml` | chunk_size 200、overlap 20、检索 k=3 |
| `config/prompts.yml` | 三个 .txt prompt 文件路径 |
| `config/agent.yml` | `external_data_path` |
| `config/pipeline.yml` | Pipeline 参数、模型路径、分类名映射、标注 CSV 路径、conda 环境名 |

## Import 规范

项目不使用 `__init__.py`。每个模块在文件头部将项目根目录注入 `sys.path`：

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

然后用裸包名 import：`from model.factory import chat_model`。不要改为相对导入。

## 踩坑清单

- **两个独立 conda 环境**：agent 和 pytorch_env。Pipeline 模块必须在 pytorch_env 中运行，不能混用。
- **`agent/tools/agent_tools.py` 是空的** — 实际工具在 `recognize_ship_tool.py` 和 `rag_search_tool.py`。
- **Import 副作用无处不在**：`model/factory.py` 在 import 时创建 `chat_model` 和 `embed_model` 单例。`utils/config_handler.py` 在 import 时加载所有 YAML。`utils/logger_handler.py` 在 import 时创建 `logger` 单例。
- **`.gitignore` 屏蔽了所有 `.md` 和 `.json`**（第 97-98 行、132-133 行）。CLAUDE.md、`.claude/` 设置、prompt 文件、聊天历史 JSON 都无法正常提交，需用 `git add -f` 或修改 `.gitignore`。
- **没有测试** — 改完代码后运行 `python pipeline/pipeline_engine.py <wav_path>` 验证 Pipeline，或 `streamlit run app.py` 验证 Agent。
- **`config/.env` 被 gitignore**，必须手动创建。
- **流式输出有 10ms 人为延迟**：`app.py` 中每字符 `time.sleep(0.01)`，纯粹为了 UI 视觉效果。
- **RAG 去重依赖 `md5.text`**：`vector_store.py` 在 `data/` 同级目录维护已摄入文档 MD5 清单。若被删除，下次运行会全量重新入库。
- **`rag_service.py` 的 LCEL 链中有 `print_prompt`**，每次调用向 stdout 打印完整 prompt。这是调试残留。
