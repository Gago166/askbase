import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.core.log import logger


async def validate_sql(state: AgentState, llm) -> dict:
    """校验生成的 SQL 是否正确"""
    sql = state["sql"]
    extra_context = state.get("extra_context", "")
    question = state["question"]

    system_prompt = """你是一个严格的 SQL 校验专家。请校验给定的 SQL 语句是否正确。

校验维度：
1. 语法是否正确
2. 表名和字段名是否存在（根据提供的表结构）
3. JOIN 条件是否正确
4. WHERE 条件是否合理
5. GROUP BY 是否与 SELECT 中的非聚合字段匹配
6. 聚合函数使用是否正确
7. 是否有潜在的性能问题（如缺少索引的字段做过滤）

输出格式：
```json
{
  "valid": true/false,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}
```

只输出 JSON，不要包含其他内容。"""

    user_prompt = f"""用户问题：{question}

表结构信息：
{extra_context}

待校验的 SQL：
{sql}"""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    content = response.content.strip()
    # 解析 JSON
    import json
    try:
        result = json.loads(re.sub(r'^```json\s*|```$', '', content).strip())
    except json.JSONDecodeError:
        # 尝试提取 JSON
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            # 降级：假设 SQL 有效
            logger.warning(f"Could not parse validation result: {content}")
            return {"sql_valid": True, "validation_feedback": ""}

    is_valid = result.get("valid", True)
    issues = result.get("issues", [])
    suggestions = result.get("suggestions", [])

    feedback = ""
    if issues:
        feedback = "问题：\n" + "\n".join(f"- {i}" for i in issues)
    if suggestions:
        feedback += "\n\n建议：\n" + "\n".join(f"- {s}" for s in suggestions)

    logger.info(f"SQL validation: valid={is_valid}, feedback={feedback}")

    return {
        "sql_valid": is_valid,
        "validation_feedback": feedback,
    }
