import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState


async def filter_table(state: AgentState, llm) -> dict:
    """使用 LLM 过滤不相关的表"""
    question = state["question"]
    merged_info = state.get("merged_info", "")
    recalled_columns = state.get("recalled_columns", [])

    if not recalled_columns:
        return {"relevant_tables": [], "relevant_columns": []}

    # 提取唯一表信息
    tables_map = {}
    for col in recalled_columns:
        table_id = col.get("table_id", "unknown")
        if table_id not in tables_map:
            tables_map[table_id] = {
                "name": table_id,
                "columns": []
            }
        tables_map[table_id]["columns"].append(col)

    tables_json = json.dumps(list(tables_map.values()), ensure_ascii=False, indent=2)

    system_prompt = """你是一个数据分析专家。根据用户问题，判断需要查询哪些数据表。

要求：
1. 分析用户问题，确定需要用到哪些表
2. 事实表（fact_order）通常需要配合维度表使用
3. 输出格式为 JSON 数组，包含需要的表名，如：["fact_order", "dim_region", "dim_date"]
4. 只输出 JSON 数组，不要包含其他内容"""

    user_prompt = f"""用户问题：{question}

候选表及字段信息：
{tables_json}

请筛选出用户问题需要用到的表："""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    content = response.content.strip()
    try:
        relevant_table_names = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            relevant_table_names = json.loads(match.group())
        else:
            relevant_table_names = []

    # 筛选相关表和字段
    relevant_tables = [
        {"name": name, "columns": tables_map.get(name, {}).get("columns", [])}
        for name in relevant_table_names
        if name in tables_map
    ]
    relevant_columns = [col for col in recalled_columns if col.get("table_id") in relevant_table_names]

    return {"relevant_tables": relevant_tables, "relevant_columns": relevant_columns}
