from langchain_openai import ChatOpenAI

from app.conf.app_config import app_config


def create_llm() -> ChatOpenAI:
    """创建 LLM 实例"""
    return ChatOpenAI(
        model=app_config.llm.model_name,
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
        temperature=0,
        streaming=True,
    )
