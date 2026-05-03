import sys
import threading
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from agent.react_agent import ReactAgent
from rag.file_chat_history_store import FileChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

HISTORY_FILE = FileChatMessageHistory("001.json", "rag/chat_history")

st.set_page_config(page_title="水声目标识别智能Agent")


def on_stop():
    st.session_state["_stop_requested"] = True


# 侧边栏：文件上传
with st.sidebar:
    st.header("音频上传")
    uploaded_file = st.file_uploader("上传 WAV 音频文件", type=["wav"], label_visibility="collapsed")
    if uploaded_file is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "uwa_uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / uploaded_file.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_audio_path"] = str(tmp_path)
        st.success(f"已上传: {uploaded_file.name}")
        st.caption('在对话中说"识别这个文件"即可开始分析')
    elif "uploaded_audio_path" in st.session_state:
        st.info(f"当前文件: {Path(st.session_state['uploaded_audio_path']).name}")

    st.divider()
    if st.button("清空对话历史"):
        HISTORY_FILE.clear()
        st.session_state["message"] = []
        st.session_state["session_message"] = []
        st.success("历史记录已清空！")
        st.rerun()

# 标题
st.title("水声目标识别智能Agent")

# 主区域
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    loaded = []
    for msg in HISTORY_FILE.messages:
        if isinstance(msg, HumanMessage):
            loaded.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            loaded.append({"role": "assistant", "content": msg.content})
    st.session_state["message"] = loaded

if "session_message" not in st.session_state:
    st.session_state["session_message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# Stop button inline above chat input — only visible during generation
stop_area = st.empty()

# 用户输入
prompt = st.chat_input()

if prompt:
    # If user asks to identify but hasn't uploaded, hint the audio path
    augmented_prompt = prompt
    if "uploaded_audio_path" in st.session_state:
        audio_path = st.session_state["uploaded_audio_path"]
        if any(kw in prompt.lower() for kw in ["识别", "分析", "identify", "analyze", "这个文件", "这段音频"]):
            augmented_prompt = f"音频路径: {audio_path}\n\n用户请求: {prompt}"

    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    st.session_state["session_message"].append({"role": "user", "content": augmented_prompt})

    # Show stop button inline
    stop_area.button("停止生成", on_click=on_stop, key="stop_inline", use_container_width=True)

    # Capture locals BEFORE defining _run — st.session_state must NOT be accessed from bg thread
    st.session_state["_stop_requested"] = False
    _agent = st.session_state["agent"]
    _history = list(st.session_state["session_message"][:-1])
    response_chunks = []
    agent_error = [None]
    stop_event = threading.Event()

    def _run():
        try:
            gen = _agent.execute_stream(augmented_prompt, history=_history)
            first_chunk = True
            for chunk in gen:
                if stop_event.is_set():
                    break
                if first_chunk and chunk.strip() == augmented_prompt:
                    first_chunk = False
                    continue
                if "Question:" not in chunk:
                    response_chunks.append(chunk)
        except Exception as e:
            agent_error[0] = str(e)

    t = threading.Thread(target=_run)
    t.start()

    # Poll for results
    output_area = st.empty()
    last_len = 0
    with st.spinner("Agent思考中..."):
        while t.is_alive():
            if st.session_state.get("_stop_requested"):
                _agent.stop()
                stop_event.set()
            if len(response_chunks) > last_len:
                output_area.chat_message("assistant").markdown("".join(response_chunks))
                last_len = len(response_chunks)
            t.join(timeout=0.1)

    t.join(timeout=5)

    clean_response = "".join(response_chunks)
    if agent_error[0]:
        output_area.chat_message("assistant").markdown(f"*[错误: {agent_error[0]}]*")
    elif clean_response:
        output_area.chat_message("assistant").markdown(clean_response)
        st.session_state["message"].append({"role": "assistant", "content": clean_response})
        st.session_state["session_message"].append({"role": "assistant", "content": clean_response})
        HISTORY_FILE.add_messages([
            HumanMessage(content=prompt),
            AIMessage(content=clean_response)
        ])
    else:
        output_area.chat_message("assistant").markdown("*[已停止]*")

    stop_area.empty()
    st.rerun()