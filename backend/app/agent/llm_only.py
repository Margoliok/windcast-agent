"""H3 тексеруге арналған базалық жүйе: құралсыз LLM (LLM-only).

LLM-ге сұрақпен бірге «шикі» деректер беріледі (Rojas et al. әдістемесі): станция паспорты,
нормаланған қуат қисығы, 72 сағаттық ауа райы болжамы, Kelmarsh үшін айлық өндіріс
қорытындысы. Сандарды LLM өзі есептейді — құралдар мен тексеру шлюзі жоқ.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_model, usage_tokens
from app.core.farms import get_farm
from app.models.registry import load_bundle
from app.services import forecast_service as fs

SYSTEM = {
    "kk": "Сен — жел электр станциясы бойынша көмекшісің. Төменде берілген деректерді пайдаланып, сұраққа "
          "қазақ тілінде қысқа әрі нақты жауап бер. Өлшем бірліктері: МВт, МВт·сағ, м/с.",
    "ru": "Ты — помощник по ветровой электростанции. Используя данные ниже, кратко и точно ответь на вопрос "
          "на русском языке. Единицы: МВт, МВт·ч, м/с.",
}


def context_text(farm_id: str) -> tuple[str, dict]:
    """Промптқа берілетін деректер мәтіні және оның сандары (faithfulness үшін)."""
    farm = get_farm(farm_id)
    b = load_bundle(farm_id)
    grid = np.arange(0, 26, 1.0)
    curve = {f"{v:.0f}": round(float(c), 3) for v, c in zip(grid, b.curve(grid))}
    wx = fs.weather_forecast(farm_id, 72)
    lines = [
        f"Станция: {farm.name}; орнатылған қуат {farm.capacity_mw} МВт; ступица биіктігі {farm.hub_height_m} м; "
        f"уақыт белдеуі {farm.timezone}. Қазіргі жергілікті уақыт: {fs.fmt_local(fs.now_utc(), farm)}.",
        "Нормаланған қуат қисығы (жел м/с -> P/P_inst): " + ", ".join(f"{k}:{v}" for k, v in curve.items()),
        "72 сағаттық ауа райы болжамы (жергілікті уақыт; ступица биіктігіндегі жел м/с; 100 м жел м/с; температура °C):",
    ]
    lines += [f"{r['time_local']}; {r['ws_hub']}; {r['ws100']}; {r['t2m']}" for r in wx["series"]]
    data = {"curve": curve, "weather": wx}
    if farm.has_scada:
        from app.services.history_service import _load_history, data_range

        rng = data_range(farm_id)
        h = _load_history(farm_id, fs.parse_local(rng["start_local"], farm), fs.parse_local(rng["end_local"], farm) + pd.Timedelta(hours=1))
        m = h.raw_power_mw.clip(lower=0).groupby(h.index.to_period("M")).agg(["sum", "mean", "max"]).round(2)
        lines.append(f"Тарихи өндіріс (айлық қорытынды, {rng['start_local']} – {rng['end_local']}): ай; энергия МВт·сағ; орташа МВт; максимум МВт")
        lines += [f"{p}; {r['sum']}; {r['mean']}; {r['max']}" for p, r in m.iterrows()]
        data["history_monthly"] = m.reset_index(drop=True).to_dict(orient="records")
    return "\n".join(lines), data


def ask_llm_only(question: str, farm_id: str = "kelmarsh", lang: str = "kk") -> dict:
    t0 = time.time()
    ctx, data = context_text(farm_id)
    model = get_chat_model()
    msg = model.invoke([SystemMessage(content=SYSTEM.get(lang, SYSTEM["kk"])),
                        HumanMessage(content=f"ДЕРЕКТЕР:\n{ctx}\n\nСҰРАҚ: {question}")])
    return {"answer": msg.content or "", "latency_s": round(time.time() - t0, 2), "tokens": usage_tokens(msg),
            "context_data": data, "tools_called": [], "status": "ok"}
