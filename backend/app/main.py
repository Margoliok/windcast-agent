"""WindCast Agent — FastAPI қосымшасы."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import agent, core
from app.core.db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("fontTools", "httpx", "httpcore", "apscheduler.executors.default"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app import scheduler
    from app.services.db_sync import sync_all

    init_db()
    try:
        sync_all()
    except Exception as e:  # деректер әлі дайын болмауы мүмкін
        logging.warning("DB sync skipped: %s", e)
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="WindCast Agent API", version="1.0.0", lifespan=lifespan,
              description="ЖЭС қуатын ауа райы деректері негізінде болжайтын ЖИ-агент платформасы")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(core.router)
app.include_router(agent.router)
