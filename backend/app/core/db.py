"""Дерекқорға қосылу (SQLAlchemy 2). Әдепкі — SQLite, Docker-де — PostgreSQL."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        url = get_settings().db_url
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        _engine = create_engine(url, **kw)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


@contextmanager
def session() -> Session:
    engine()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    from app.db.models import Base

    Base.metadata.create_all(engine())
