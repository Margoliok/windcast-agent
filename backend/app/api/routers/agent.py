"""ЖИ-агентпен чат: синхронды жауап және SSE ағыны (траектория қадам-қадаммен)."""
from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sse_starlette.sse import EventSourceResponse

from app.agent import graph
from app.core.db import session
from app.db.models import AgentLog

router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    lang: str = "kk"
    farm_id: str = "kelmarsh"
    session_id: str | None = None
    history: list[dict] = []


@router.post("/chat")
def chat(req: ChatRequest):
    return graph.ask(req.question, req.lang, req.farm_id, req.session_id, req.history)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def gen():
        for ev in graph.stream(req.question, req.lang, req.farm_id, req.session_id, req.history):
            yield {"event": ev["event"], "data": json.dumps(ev["data"], ensure_ascii=False, default=str)}

    return EventSourceResponse(gen())


@router.get("/logs")
def logs(limit: int = 50):
    with session() as s:
        rows = s.execute(select(AgentLog).order_by(desc(AgentLog.id)).limit(limit)).scalars().all()
        return [{"id": r.id, "session_id": r.session_id, "created_at": r.created_at, "question": r.question,
                 "tools": r.tools_called, "faithfulness": r.faithfulness, "latency_s": r.latency_s,
                 "tokens": r.tokens, "system": r.system, "answer": r.answer[:500]} for r in rows]
