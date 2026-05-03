import os
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_tool import get_abs_path

import dotenv
dotenv.load_dotenv(get_abs_path("config/.env"))
os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("LLM_BASE_URL")

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage, AIMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools.recognize_ship_tool import recognize_ship
from agent.tools.rag_search_tool import rag_search

_TOOL_PROGRESS = {
    "recognize_ship": "正在分析音频特征并运行深度学习识别模型...",
    "rag_search": "正在检索水声领域知识库...",
}

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[recognize_ship, rag_search],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def execute_stream(self, query: str, history: list = None):
        self._stop_event.clear()
        messages = list(history) if history else []
        messages.append({"role": "user", "content": query})
        input_dict = {
            "messages": messages
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            if self._stop_event.is_set():
                break
            latest = chunk["messages"][-1]

            # Skip user message echo
            if latest.type == "human":
                continue

            # Skip raw tool results (JSON full of numbers)
            if isinstance(latest, ToolMessage):
                continue

            # Tool call decision → friendly progress message
            if isinstance(latest, AIMessage) and latest.tool_calls:
                for tc in latest.tool_calls:
                    name = tc.get("name", "")
                    msg = _TOOL_PROGRESS.get(name, f"正在执行 {name}...")
                    yield msg + "\n"
                continue

            # Final assistant response
            if latest.content:
                yield latest.content.strip() + "\n"