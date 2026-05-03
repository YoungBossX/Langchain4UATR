"""
LangChain tool: recognize_ship — invoke the recognition pipeline via subprocess.
"""
import sys
import json
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.tools import tool
from utils.path_tool import get_project_root
from utils.config_handler import pipeline_conf


@tool
def recognize_ship(audio_path: str) -> dict:
    """对上传的水声录音进行目标识别，返回结构化识别结果。

    包含音频信息、环境参数、每段预测（类别、置信度、Top-5概率）、
    聚合结果（多数投票、平均置信度）、模型内部信息（跨模态注意力权重等）。

    Args:
        audio_path: 音频文件的绝对路径（wav格式，16kHz单声道最佳）
    """
    project_root = str(get_project_root())
    pipeline_script = str(Path(project_root) / "pipeline" / "pipeline_engine.py")
    py_conf = pipeline_conf["pytorch_env"]

    proc = subprocess.run(
        [
            "conda", "run", "-n", py_conf["name"],
            "python", pipeline_script, audio_path,
        ],
        capture_output=True,
        text=True,
        timeout=py_conf["timeout_seconds"],
        cwd=project_root,
    )

    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip().split("\n")[-10:]
        err_msg = "\n".join(stderr_tail) if stderr_tail else proc.stderr[:500]
        return {
            "status": "error",
            "error_message": f"Pipeline 执行失败 (returncode={proc.returncode}):\n{err_msg[:500]}",
        }

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_message": f"Pipeline 输出解析失败:\n{proc.stdout[:500]}",
        }
