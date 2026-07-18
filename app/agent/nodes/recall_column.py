from app.agent.state import AgentState


async def recall_column(state: AgentState, runtime: dict) -> dict:
    """通过 Qdrant 向量搜索召回相关字段信息"""
    keywords = state["keywords"]
    question = state["question"]

    if not keywords:
        return {"recalled_columns": []}

    embedding_client = runtime["embedding_client"]
    column_qdrant_repo = runtime["column_qdrant_repo"]

    # 用问题全文和关键词分别查询
    all_columns = []
    seen_ids = set()

    # 用问题全文搜索
    question_embedding = await embedding_client.aembed_query(question)
    question_results = await column_qdrant_repo.search(question_embedding, limit=10, score_threshold=0.3)
    for point in question_results:
        if point.id not in seen_ids:
            all_columns.append(point.payload)
            seen_ids.add(point.id)

    # 用每个关键词搜索
    for keyword in keywords[:5]:  # 限制前5个关键词
        kw_embedding = await embedding_client.aembed_query(keyword)
        kw_results = await column_qdrant_repo.search(kw_embedding, limit=5, score_threshold=0.3)
        for point in kw_results:
            if point.id not in seen_ids:
                all_columns.append(point.payload)
                seen_ids.add(point.id)

    return {"recalled_columns": all_columns}
