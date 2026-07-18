from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph 问数智能体状态"""
    # 用户输入
    question: str

    # 关键词抽取
    keywords: list[str]

    # 召回结果
    recalled_columns: list[dict[str, Any]]
    recalled_metrics: list[dict[str, Any]]
    recalled_values: list[dict[str, Any]]

    # 合并后的召回信息
    merged_info: str

    # 过滤后的相关信息
    relevant_columns: list[dict[str, Any]]
    relevant_metrics: list[dict[str, Any]]
    relevant_tables: list[dict[str, Any]]
    relevant_values: list[dict[str, Any]]

    # 补充上下文
    extra_context: str

    # SQL 相关
    sql: str
    sql_valid: bool
    validation_feedback: str
    sql_retry_count: int
    max_sql_retries: int

    # 执行结果
    query_result: Optional[list[dict[str, Any]]]
    error_message: str

    # 流式进度
    progress_messages: list[dict[str, str]]
