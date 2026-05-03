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