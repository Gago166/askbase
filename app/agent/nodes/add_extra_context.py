from app.agent.state import AgentState


async def add_extra_context(state: AgentState, llm, runtime: dict) -> dict:
    """添加额外上下文信息（如表结构详情、关联关系等）"""
    relevant_tables = state.get("relevant_tables", [])
    relevant_columns = state.get("relevant_columns", [])
    relevant_metrics = state.get("relevant_metrics", [])

    if not relevant_tables and not relevant_columns:
        return {"extra_context": ""}

    # 构建结构化的上下文
    context_parts = []

    # 按表组织字段
    table_columns: dict[str, list] = {}
    for col in relevant_columns:
        tid = col.get("table_id", "unknown")
        if tid not in table_columns:
            table_columns[tid] = []
        table_columns[tid].append(col)

    context_parts.append("## 可用表结构\n")

    for table in relevant_tables:
        tname = table.get("name", "")
        cols = table_columns.get(tname, [])
        if cols:
            context_parts.append(f"### 表: {tname}")
            context_parts.append("```sql")
            cols_def = []
            for col in cols:
                nullable = "NOT NULL" if col.get("role") == "primary_key" else ""
                cols_def.append(f"  {col.get('name')} {col.get('type')} {nullable}  -- {col.get('description')}")
            context_parts.append(f"CREATE TABLE {tname} (\n" + ",\n".join(cols_def) + "\n);")
            context_parts.append("```\n")

    # 添加关联关系说明
    context_parts.append("## 表关联关系")
    context_parts.append("- fact_order.customer_id → dim_customer.customer_id")
    context_parts.append("- fact_order.product_id → dim_product.product_id")
    context_parts.append("- fact_order.date_id → dim_date.date_id")
    context_parts.append("- fact_order.region_id → dim_region.region_id\n")

    # 添加指标定义
    if relevant_metrics:
        context_parts.append("## 可用指标定义")
        for metric in relevant_metrics:
            context_parts.append(f"- **{metric.get('name')}**: {metric.get('description')}")
            if metric.get('relevant_columns'):
                context_parts.append(f"  关联字段: {', '.join(metric['relevant_columns'])}")
        context_parts.append("")

    extra_context = "\n".join(context_parts)

    return {"extra_context": extra_context}
