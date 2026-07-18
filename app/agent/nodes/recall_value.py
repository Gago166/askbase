from app.agent.state import AgentState


async def recall_value(state: AgentState, runtime: dict) -> dict:
    """通过 ES 全文搜索召回相关字段取值"""
    keywords = state["keywords"]

    if not keywords:
        return {"recalled_values": []}

    value_es_repo = runtime["value_es_repo"]

    all_values = []
    seen_ids = set()

    # 用每个关键词在 ES 中搜索
    for keyword in keywords:
        try:
            results = await value_es_repo.search(keyword, limit=10)
            for hit in results:
                source = hit["_source"]
                if source["id"] not in seen_ids:
                    all_values.append(source)
                    seen_ids.add(source["id"])
        except Exception:
            continue

    return {"recalled_values": all_values}
