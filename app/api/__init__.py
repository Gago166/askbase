import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.agent.graph import build_graph
from app.agent.llm import create_llm
from app.agent.state import AgentState
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import dw_mysql_client_manager, meta_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.core.log import logger
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.repositories.es.value_es_repository import ValueESRepository


# ---- 应用工厂 ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化所有无状态客户端
    logger.info("智源问数 启动中...")
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()
    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()
    logger.info("所有客户端初始化完成")

    yield

    # 关闭时清理资源
    await meta_mysql_client_manager.close()
    await dw_mysql_client_manager.close()
    await qdrant_client_manager.close()
    await es_client_manager.close()
    logger.info("智源问数 已关闭")


app = FastAPI(
    title="智源问数",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- 请求/响应模型 ----
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    type: str
    data: list[dict] | None = None
    message: str | None = None


# ---- 缓存图结构 ----
_graph = None
_llm = None


def _get_or_build_graph():
    """获取或构建图结构（结构可复用，运行时上下文每次请求新建）"""
    global _graph, _llm
    if _graph is None:
        _llm = create_llm()
        _graph = build_graph()  # 图结构，不含 LLM 和 context 绑定
    return _graph, _llm


# ---- 辅助函数 ----
_STEP_NAMES = {
    "extract_keywords": "抽取关键词",
    "recall_column": "召回字段",
    "recall_metric": "召回指标",
    "recall_value": "召回字段取值",
    "merge_retrieved_info": "合并召回信息",
    "filter_metric": "过滤指标",
    "filter_table": "过滤表格",
    "add_extra_context": "补充上下文",
    "generate_sql": "生成SQL",
    "validate_sql": "校验SQL",
    "correct_sql": "校正SQL",
    "execute_sql": "执行SQL",
}


def _make_runtime_config():
    """创建请求级别的运行时配置（包含数据库会话）"""
    return {
        "configurable": {
            "llm": _llm,
            "embedding_client": embedding_client_manager.client,
            "column_qdrant_repo": ColumnQdrantRepository(qdrant_client_manager.client),
            "metric_qdrant_repo": MetricQdrantRepository(qdrant_client_manager.client),
            "value_es_repo": ValueESRepository(es_client_manager.client),
            "meta_session_factory": meta_mysql_client_manager.session_factory,
            "dw_session_factory": dw_mysql_client_manager.session_factory,
        }
    }


# ---- API 路由 ----
@app.get("/")
async def index():
    """前端页面"""
    from pathlib import Path
    static_dir = Path(__file__).parent.parent / "static" / "index.html"
    return FileResponse(static_dir)


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """同步问数接口"""
    graph, _ = _get_or_build_graph()
    config = _make_runtime_config()
    initial_state = AgentState(question=request.question)

    try:
        final_state = await graph.ainvoke(initial_state, config=config)

        if final_state.get("error_message"):
            return QueryResponse(
                type="error",
                message=final_state["error_message"]
            )

        return QueryResponse(
            type="result",
            data=final_state.get("query_result", [])
        )
    except Exception as e:
        logger.error(f"问数失败: {str(e)}")
        return QueryResponse(type="error", message=str(e))


@app.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    """流式问数接口 — SSE 实时输出执行进度和结果"""
    graph, _ = _get_or_build_graph()
    config = _make_runtime_config()
    initial_state = AgentState(question=request.question)

    async def event_generator():
        try:
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event")

                if kind == "on_chain_start":
                    name = event.get("name", "")
                    if name in _STEP_NAMES:
                        yield f"data: {json.dumps({'type': 'progress', 'step': _STEP_NAMES[name], 'status': 'running'}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end":
                    name = event.get("name", "")
                    output = event.get("data", {}).get("output", {})

                    if name == "execute_sql":
                        if isinstance(output, dict):
                            if output.get("error_message"):
                                yield f"data: {json.dumps({'type': 'error', 'message': output['error_message']}, ensure_ascii=False)}\n\n"
                            elif output.get("query_result") is not None:
                                yield f"data: {json.dumps({'type': 'result', 'data': output['query_result']}, ensure_ascii=False)}\n\n"

                    # 进度消息
                    if isinstance(output, dict) and "progress_messages" in output:
                        for msg in output["progress_messages"]:
                            yield f"data: {json.dumps({'type': 'progress', 'step': msg['step'], 'status': msg['status']}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式问数失败: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
