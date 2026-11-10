"""OpenAI GPT клиенті (langchain-openai)."""
from __future__ import annotations

from app.core.config import get_settings


class LLMNotConfigured(RuntimeError):
    pass


def get_chat_model():
    s = get_settings()
    if not s.openai_api_key:
        raise LLMNotConfigured("OPENAI_API_KEY көрсетілмеген (.env файлына қосыңыз)")
    from langchain_openai import ChatOpenAI

    kw = {"model": s.openai_model, "api_key": s.openai_api_key, "timeout": 120, "max_retries": 2}
    if s.openai_base_url:
        kw["base_url"] = s.openai_base_url
    return ChatOpenAI(**kw)


def usage_tokens(msg) -> int:
    u = getattr(msg, "usage_metadata", None) or {}
    return int(u.get("total_tokens", 0) or 0)
