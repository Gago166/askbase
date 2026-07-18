import json
from typing import Literal

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes.extract_keywords import extract_keywords
from app.agent.nodes.recall_column import recall_column
from app.agent.nodes.recall_metric import recall_metric
from app.agent.nodes.recall_value import recall_value
from app.agent.nodes.merge_retrieved_info import merge_retrieved_info
from app.agent.nodes.filter_metric import filter_metric
from app.agent.nodes.filter_table import filter_table
from app.agent.nodes.add_extra_context import add_extra_context
from app.agent.nodes.generate_sql import generate_sql
from app.agent.nodes.validate_sql import validate_sql
from app.agent.nodes.correct_sql import correct_sql
from app.agent.nodes.execute_sql import execute_sql
from app.core.log import logger


def _get_llm(config):
    """从运行时配置中获取 LLM"""
    return config["configurable"]["llm"]


def _get_runtime(config):
    """从运行时配置中获取所有资源"""
    return config["configurable"]


# ---- 节点包装（从 config 中提取运行时资源）----

async def _extract_keywords_node(state: AgentState, config) -> dict:
    llm = _get_llm(config)
    return await extract_keywords(state, llm)


async def _recall_column_node(state: AgentState, config) -> dict:
    rt = _get_runtime(config)
    return await recall_column(state, rt)


async def _recall_metric_node(state: AgentState, config) -> dict:
    rt = _get_runtime(config)
    return await recall_metric(state, rt)


async def _recall_value_node(state: AgentState, config) -> dict:
    rt = _get_runtime(config)
    return await recall_value(state, rt)


async def _merge_retrieved_info_node(state: AgentState) -> dict:
    return await merge_retrieved_info(state)


async def _filter_metric_node(state: AgentState, config) -> dict:
    llm = _get_llm(config)
    return await filter_metric(state, llm)


async def _filter_table_node(state: AgentState, config) -> dict:
    llm = _get_llm(config)
    return await filter_table(state, llm)


async def _add_extra_context_node(state: AgentState, config) -> dict:
    llm = _get_llm(config)
    rt = _get_runtime(config)
    return await add_extra_context(state, llm, rt)


async def _generate_sql_node(state: AgentState, config) -> dict:
    llm = _get_llm(config)
    return await generate_sql(state, llm)


async def _validate_sql_node(state: AgentState, config) -> dict:
    llm = _get_llm(config)
    return await validate_sql(state, llm)


async def _correct_sql_node(state: AgentState, config) -> dict:
    llm = _get_llm(config)
    return await correct_sql(state, llm)


async def _execute_sql_node(state: AgentState, config) -> dict:
    rt = _get_runtime(config)
    return await execute_sql(state, rt)


def _should_retry_sql(state: AgentState) -> Literal["correct_sql", "execute_sql"]:
    """判断是否需要修正重试"""
    if state.get("sql_valid", False):
        return "execute_sql"
    if state.get("sql_retry_count", 0) >= state.get("max_sql_retries", 3):
        logger.warning(f"SQL 校验失败已达最大重试次数 {state.get('max_sql_retries', 3)}，直接执行")
        return "execute_sql"
    return "correct_sql"


def build_graph() -> StateGraph:
    """构建 LangGraph 问数工作流"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("extract_keywords", _extract_keywords_node)
    workflow.add_node("recall_column", _recall_column_node)
    workflow.add_node("recall_metric", _recall_metric_node)
    workflow.add_node("recall_value", _recall_value_node)
    workflow.add_node("merge_retrieved_info", _merge_retrieved_info_node)
    workflow.add_node("filter_metric", _filter_metric_node)
    workflow.add_node("filter_table", _filter_table_node)
    workflow.add_node("add_extra_context", _add_extra_context_node)
    workflow.add_node("generate_sql", _generate_sql_node)
    workflow.add_node("validate_sql", _validate_sql_node)
    workflow.add_node("correct_sql", _correct_sql_node)
    workflow.add_node("execute_sql", _execute_sql_node)

    # 入口
    workflow.set_entry_point("extract_keywords")

    # 关键词 → 三路并行召回
    workflow.add_edge("extract_keywords", "recall_column")
    workflow.add_edge("extract_keywords", "recall_metric")
    workflow.add_edge("extract_keywords", "recall_value")

    # 三路召回 → 合并
    workflow.add_edge("recall_column", "merge_retrieved_info")
    workflow.add_edge("recall_metric", "merge_retrieved_info")
    workflow.add_edge("recall_value", "merge_retrieved_info")

    # 合并 → 过滤指标 → 过滤表 → 补充上下文
    workflow.add_edge("merge_retrieved_info", "filter_metric")
    workflow.add_edge("filter_metric", "filter_table")
    workflow.add_edge("filter_table", "add_extra_context")

    # 上下文 → 生成SQL → 校验
    workflow.add_edge("add_extra_context", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")

    # 校验 → 条件：通过→执行，失败→修正
    workflow.add_conditional_edges(
        "validate_sql",
        _should_retry_sql,
        {"correct_sql": "correct_sql", "execute_sql": "execute_sql"}
    )

    # 修正 → 重新校验
    workflow.add_edge("correct_sql", "validate_sql")

    # 执行 → 结束
    workflow.add_edge("execute_sql", END)

    return workflow.compile()
