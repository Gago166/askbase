from app.agent.state import AgentState


async def recall_metric(state: AgentState, runtime: dict) -> dict:
    """通过 Qdrant 向量搜索召回相关指标信息"""
    keywords = state["keywords"]
    question = state["question"]

    if not keywords:
        return {"recalled_metrics": []}

    embedding_client = runtime["embedding_client"]
    metric_qdrant_repo = runtime["metric_qdrant_repo"]

    all_metrics = []
    seen_ids = set()

    # 用问题全文搜索
    question_embedding = await embedding_client.aembed_query(question)
    question_results = await metric_qdrant_repo.search(question_embedding, limit=10, score_threshold=0.3)
    for point in question_results:
        if point.id not in seen_ids:
            all_metrics.append(point.payload)
            seen_ids.add(point.id)

    # 用每个关键词搜索
    for keyword in keywords[:5]:
        kw_embedding = await embedding_client.aembed_query(keyword)
        kw_results = await metric_qdrant_repo.search(kw_embedding, limit=5, score_threshold=0.3)
        for point in kw_results:
            if point.id not in seen_ids:
                all_metrics.append(point.payload)
                seen_ids.add(point.id)

    return {"recalled_metrics": all_metrics}
