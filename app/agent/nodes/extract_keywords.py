import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState


async def extract_keywords(state: AgentState, llm) -> dict:
    """从用户问题中抽取关键词"""
    question = state["question"]

    system_prompt = """你是一个专业的数据分析关键词抽取助手。你的任务是从用户的自然语言问题中提取出用于检索数据库表、字段和指标的关键词。

要求：
1. 提取的关键词应该是与数据查询相关的核心词汇
2. 包括：指标类词汇（如销售额、销量）、维度类词汇（如地区、品类、时间）、实体类词汇（如客户、订单、商品）
3. 保留原始表达，同时补充同义表达
4. 输出格式为 JSON 数组，如：["关键词1", "关键词2", ...]
5. 只输出 JSON 数组，不要包含其他内容"""

    user_prompt = f"用户问题：{question}\n\n请提取关键词："

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # 解析 LLM 返回的关键词列表
    content = response.content.strip()
    try:
        # 尝试直接解析 JSON
        keywords = json.loads(content)
    except json.JSONDecodeError:
        # 尝试从文本中提取 JSON 数组
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            keywords = json.loads(match.group())
        else:
            # 降级：按逗号分割
            keywords = [k.strip() for k in content.strip('[]"\' ').split(',') if k.strip()]

    return {"keywords": keywords}
