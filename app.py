import sys
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
st.divider()

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

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    st.session_state["session_message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("Agent思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            prompt,
            history=st.session_state["session_message"][:-1] 
        )

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.session_state["session_message"].append({"role": "assistant", "content": response_messages[-1]})

        HISTORY_FILE.add_messages([
            HumanMessage(content=prompt),
            AIMessage(content=response_messages[-1])
        ])
        st.rerun()