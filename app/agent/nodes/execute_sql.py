from sqlalchemy import text

from app.agent.state import AgentState
from app.core.log import logger
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository


async def execute_sql(state: AgentState, runtime: dict) -> dict:
    """执行 SQL 查询并返回结果"""
    sql = state["sql"]

    if not sql:
        return {"query_result": None, "error_message": "SQL 为空，无法执行"}

    dw_session_factory = runtime.get("dw_session_factory")
    if not dw_session_factory:
        return {"query_result": None, "error_message": "数据库会话工厂未初始化"}

    try:
        async with dw_session_factory() as session:
            dw_repo = DWMySQLRepository(session)
            result = await session.execute(text(sql))
            rows = result.fetchall()

            # 转换为字典列表
            columns = list(result.keys())
            data = []
            for row in rows:
                data.append(dict(zip(columns, [str(v) if v is not None else None for v in row])))

            logger.info(f"SQL executed successfully, {len(data)} rows returned")
            return {"query_result": data, "error_message": ""}

    except Exception as e:
        error_msg = f"SQL 执行失败：{str(e)}"
        logger.error(error_msg)
        return {"query_result": None, "error_message": error_msg}
