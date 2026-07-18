import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.core.log import logger


async def generate_sql(state: AgentState, llm) -> dict:
    """生成 SQL 查询语句"""
    question = state["question"]
    extra_context = state.get("extra_context", "")
    merged_info = state.get("merged_info", "")

    system_prompt = """你是一个专业的 SQL 生成助手。根据用户问题、表结构和元数据信息，生成正确的 MySQL SQL 查询。

要求：
1. 仔细分析用户问题，理解查询意图
2. 根据表结构选择合适的表和字段
3. 正确编写 JOIN 条件
4. 正确编写 WHERE 条件和 GROUP BY
5. 使用标准 MySQL 语法
6. 只输出 SQL 语句，不要包含任何其他内容（不要用 ```sql 包裹）
7. 表名和字段名使用反引号包裹

注意事项：
- 维度表通过外键与事实表关联
- dim_date 表的 date_id 格式为 yyyyMMdd
- 金额字段 order_amount 是数值类型，可以直接 SUM/AVG
- 数量字段 order_quantity 是数值类型"""

    user_prompt = f"""用户问题：{question}

{extra_context}

{merged_info}

请生成 SQL 查询："""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    sql = response.content.strip()
    # 清理可能的 markdown 包裹
    sql = re.sub(r'^```sql\s*', '', sql)
    sql = re.sub(r'^```\s*', '', sql)
    sql = re.sub(r'\s*```$', '', sql)
    sql = sql.strip()

    logger.info(f"Generated SQL: {sql}")

    return {"sql": sql}
