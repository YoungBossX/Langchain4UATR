"""
LangChain tool: rag_search — search the underwater acoustic knowledge base.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService

_rag_service = None


def _get_rag():
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


@tool
def rag_search(query: str) -> str:
    """搜索水声领域知识库，返回基于参考资料的摘要回答。

    可用于查询船舶类型特征、声学特性、水下噪声来源、声传播知识等。

    Args:
        query: 中文或英文搜索查询字符串
    """
    try:
        return _get_rag().rag_summarize(query)
    except Exception as e:
        return f"知识库搜索失败: {str(e)}"
