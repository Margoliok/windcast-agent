"""9-қадам: GitHub Pages демосы үшін ЖИ-агенттің НАҚТЫ диалогтарын жазу.

Агент (OpenAI GPT + құралдар + тексеру шлюзі) әр сұраққа шын жауап береді; жауап, траектория,
шлюз нәтижесі және артефактілер reports/agent_eval/demo_recordings.json ішіне сақталады.
Кейін 08_export_demo.py оларды демоға қосады. OPENAI_API_KEY қажет.
"""
import json

import _bootstrap  # noqa: F401

from app.agent import graph
from app.core.config import get_settings
from app.core.db import init_db
from app.services.db_sync import sync_all

QUESTIONS = [
    ("kk", "kelmarsh", "Ертең 10:00-ден 18:00-ге дейін қанша энергия өндіреміз?"),
    ("kk", "kelmarsh", "Ертеңгі болжамды түсіндір: кешкі өндіріс неге өзгереді?"),
    ("kk", "kelmarsh", "Жел болжамнан 15% күшті болса, келесі 24 сағатта қанша өндіреміз?"),
    ("kk", "ereymentau", "Осы аптада жел 4 м/с-тан төмен 6 сағаттан ұзақ болатын кезеңдер бар ма?"),
    ("kk", "kelmarsh", "Тест кезеңінде модельдерді 24 сағат горизонтында салыстыр."),
    ("kk", "ereymentau", "Ертеңгі тәуліктік кестені Excel-ге шығар."),
    ("kk", "kelmarsh", "2019 жылғы қаңтарда турбиналар жел жеткілікті болса да тоқтап тұрған кезеңдерді тап."),
    ("kk", "ereymentau", "Ерейментау ЖЭС өткен айда нақты қанша энергия өндірді?"),
    ("kk", "kelmarsh", "Бүгін доллардың теңгеге бағамы қандай?"),
    ("ru", "kelmarsh", "Сколько выработаем завтра с 10:00 до 18:00?"),
    ("ru", "ereymentau", "Сформируй суточный график на завтра и выгрузи в Excel."),
    ("ru", "kelmarsh", "Насколько точной была модель на тестовом периоде? Сравни с persistence на горизонте 24 ч."),
]

if __name__ == "__main__":
    if not get_settings().openai_api_key:
        raise SystemExit("OPENAI_API_KEY жоқ — .env файлына қосыңыз.")
    init_db()
    sync_all()
    out = []
    for lang, farm, q in QUESTIONS:
        r = graph.ask(q, lang, farm)
        print(f"[{r['status']}] {q}\n  -> {r['answer'][:160]}")
        out.append({"lang": lang, "farm_id": farm, "question": q, **r})
    p = get_settings().reports_dir / "agent_eval" / "demo_recordings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Жазылды: {p}. Енді: python scripts/08_export_demo.py")
