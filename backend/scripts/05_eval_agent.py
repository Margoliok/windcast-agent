"""5-қадам: ЖИ-агентті бағалау (E6, H3): агент vs құралсыз LLM -> Кесте 17, Қосымша 3.

Іске қосу:  python scripts/05_eval_agent.py [--limit N] [--only agent|llm_only]
OPENAI_API_KEY .env файлында болуы керек. Кілтсіз тек сұрақтар тізімі (Қосымша 3) жасалады.
"""
import argparse

import _bootstrap  # noqa: F401

from app.core.config import get_settings
from app.core.db import init_db
from app.evaluation.agent_eval import export_questions, run
from app.services.db_sync import sync_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", choices=["agent", "llm_only"], default=None)
    a = ap.parse_args()
    init_db()
    sync_all()
    export_questions()
    print("Қосымша 3: reports/agent_eval/questions_appendix3.csv")
    if not get_settings().openai_api_key:
        print("OPENAI_API_KEY жоқ — агентті бағалау өткізілді. .env файлына кілтті қосып, қайта іске қосыңыз.")
    else:
        run(systems=(a.only,) if a.only else ("agent", "llm_only"), limit=a.limit)
        print(open(get_settings().reports_dir / "tables" / "tab17_agent_vs_llm.md", encoding="utf-8").read())
