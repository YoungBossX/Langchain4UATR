import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import streamlit as st
from agent.react_agent import ReactAgent
from rag.file_chat_history_store import FileChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

HISTORY_FILE = FileChatMessageHistory("001.json", "rag/chat_history")

# 标题
st.title("水声目标识别智能Agent")

# 侧边栏：文件上传
with st.sidebar:
    st.header("音频上传")
    uploaded_file = st.file_uploader("上传 WAV 音频文件", type=["wav"], label_visibility="collapsed")
    if uploaded_file is not None:
        # Save to temp file
        tmp_dir = Path(tempfile.gettempdir()) / "uwa_uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / uploaded_file.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_audio_path"] = str(tmp_path)
        st.success(f"已上传: {uploaded_file.name}")
        st.caption(f"路径: `{tmp_path}`")
        st.caption("在对话中说"识别这个文件"即可开始分析")
    elif "uploaded_audio_path" in st.session_state:
        st.info(f"当前文件: {Path(st.session_state['uploaded_audio_path']).name}")

    st.divider()
    if st.button("清空对话历史"):
        HISTORY_FILE.clear()
        st.session_state["message"] = []
        st.session_state["session_message"] = []
        st.success("历史记录已清空！")
        st.rerun()

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

    response_messages = []
    with st.spinner("Agent思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            augmented_prompt,
            history=st.session_state["session_message"][:-1]
        )

        def capture(generator, cache_list):
            first_chunk = True
            full = ""
            for chunk in generator:
                if first_chunk and chunk.strip() == augmented_prompt:
                    first_chunk = False
                    continue

                if "Question:" not in chunk:
                    cache_list.append(chunk)
                    full += chunk
                    for char in chunk:
                        time.sleep(0.01)
                        yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))

        if response_messages:
            clean_response = "".join(response_messages)
            st.session_state["message"].append({"role": "assistant", "content": clean_response})
            st.session_state["session_message"].append({"role": "assistant", "content": clean_response})

            HISTORY_FILE.add_messages([
                HumanMessage(content=prompt),
                AIMessage(content=clean_response)
            ])
        st.rerun()