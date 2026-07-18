from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """Agent 运行时上下文，包含各客户端和配置的引用"""
    # 各个 Repository 引用（在运行时注入）
    meta_repo: Any = None
    dw_repo: Any = None
    column_qdrant_repo: Any = None
    metric_qdrant_repo: Any = None
    value_es_repo: Any = None
    embedding_client: Any = None
    llm: Any = None
