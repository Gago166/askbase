import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.core.log import logger


async def correct_sql(state: AgentState, llm) -> dict:
    """根据校验反馈修正 SQL"""
    sql = state["sql"]
    feedback = state.get("validation_feedback", "")
    extra_context = state.get("extra_context", "")
    question = state["question"]
    retry_count = state.get("sql_retry_count", 0)

    system_prompt = """你是一个专业的 SQL 调优专家。根据校验反馈修正 SQL 语句。

要求：
1. 仔细分析校验反馈中提到的问题
2. 逐条修正问题
3. 保持 SQL 的原有查询意图不变
4. 只输出修正后的 SQL 语句，不要包含任何其他内容（不要用 ```sql 包裹）"""

    user_prompt = f"""用户问题：{question}

表结构信息：
{extra_context}

原 SQL：
{sql}

校验反馈：
{feedback}

请输出修正后的 SQL："""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    corrected_sql = response.content.strip()
    corrected_sql = re.sub(r'^```sql\s*', '', corrected_sql)
    corrected_sql = re.sub(r'^```\s*', '', corrected_sql)
    corrected_sql = re.sub(r'\s*```$', '', corrected_sql)
    corrected_sql = corrected_sql.strip()

    logger.info(f"Corrected SQL: {corrected_sql}")

    return {
        "sql": corrected_sql,
        "sql_retry_count": retry_count + 1,
    }
