from typing import Optional

import httpx

from app.conf.app_config import EmbeddingConfig, app_config


class TEIEmbeddingClient:
    """本地 Text Embeddings Inference (TEI) 服务的 HTTP 客户端"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    @property
    def sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=60.0, trust_env=False)
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=60.0, trust_env=False)
        return self._async_client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.sync_client.post(
            f"{self.base_url}/embed",
            json={"inputs": texts},
        )
        response.raise_for_status()
        return response.json()

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        response = await self.async_client.post(
            f"{self.base_url}/embed",
            json={"inputs": texts},
        )
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """同步批量嵌入"""
        return self._embed(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步批量嵌入"""
        return await self._aembed(texts)

    def embed_query(self, text: str) -> list[float]:
        """同步单条嵌入"""
        return self._embed([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """异步单条嵌入"""
        result = await self._aembed([text])
        return result[0]

    async def close(self):
        if self._async_client:
            await self._async_client.aclose()
        if self._sync_client:
            self._sync_client.close()


class EmbeddingClientManager:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client: Optional[TEIEmbeddingClient] = None

    def init(self):
        url = f"http://{self.config.host}:{self.config.port}"
        self.client = TEIEmbeddingClient(base_url=url)


embedding_client_manager = EmbeddingClientManager(app_config.embedding)
