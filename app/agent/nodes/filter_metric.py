import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState


async def filter_metric(state: AgentState, llm) -> dict:
    """使用 LLM 过滤不相关的指标"""
    question = state["question"]
    merged_info = state.get("merged_info", "")
    recalled_metrics = state.get("recalled_metrics", [])

    if not recalled_metrics:
        return {"relevant_metrics": []}

    metrics_json = json.dumps(recalled_metrics, ensure_ascii=False, indent=2)

    system_prompt = """你是一个数据分析专家。根据用户问题和召回的相关指标信息，判断哪些指标与用户问题真正相关。

要求：
1. 仔细分析用户问题，理解用户真正想查询的指标
2. 从候选指标中选择与问题相关的指标
3. 输出格式为 JSON 数组，包含相关指标的 name 字段，如：["GMV", "AOV"]
4. 如果都不相关，输出空数组 []
5. 只输出 JSON 数组，不要包含其他内容"""

    user_prompt = f"""用户问题：{question}

候选指标信息：
{metrics_json}

请筛选出与用户问题相关的指标："""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    content = response.content.strip()
    try:
        relevant_names = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            relevant_names = json.loads(match.group())
        else:
            relevant_names = []

    # 筛选出相关指标
    relevant_metrics = [m for m in recalled_metrics if m.get("name") in relevant_names]

    return {"relevant_metrics": relevant_metrics}
