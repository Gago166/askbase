import json

from app.agent.state import AgentState


async def merge_retrieved_info(state: AgentState) -> dict:
    """合并多路召回结果，输出结构化的元数据信息文本"""
    columns = state.get("recalled_columns", [])
    metrics = state.get("recalled_metrics", [])
    values = state.get("recalled_values", [])

    parts = []

    # 整理字段信息
    if columns:
        parts.append("## 相关字段信息")
        # 按表分组
        tables: dict[str, list] = {}
        for col in columns:
            table_id = col.get("table_id", "unknown")
            if table_id not in tables:
                tables[table_id] = []
            tables[table_id].append(col)

        for table_name, cols in tables.items():
            parts.append(f"\n### 表：{table_name}")
            for col in cols:
                parts.append(f"- 字段名: {col.get('name')}")
                parts.append(f"  类型: {col.get('type')}, 角色: {col.get('role')}")
                parts.append(f"  描述: {col.get('description')}")
                if col.get('alias'):
                    parts.append(f"  别名: {', '.join(col['alias'])}")
                if col.get('examples'):
                    parts.append(f"  示例值: {col['examples'][:5]}")

    # 整理指标信息
    if metrics:
        parts.append("\n## 相关指标信息")
        for metric in metrics:
            parts.append(f"\n- 指标名: {metric.get('name')}")
            parts.append(f"  描述: {metric.get('description')}")
            if metric.get('alias'):
                parts.append(f"  别名: {', '.join(metric['alias'])}")
            if metric.get('relevant_columns'):
                parts.append(f"  关联字段: {', '.join(metric['relevant_columns'])}")

    # 整理字段取值信息
    if values:
        parts.append("\n## 相关字段取值")
        for val in values:
            parts.append(f"- {val.get('column_id')}: {val.get('value')}")

    merged = "\n".join(parts) if parts else "未找到相关元数据信息"

    return {"merged_info": merged}
