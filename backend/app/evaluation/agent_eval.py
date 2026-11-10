"""E6: ЖИ-агентті бағалау (Кесте 17, Сурет 19, Қосымша 3) — H3 гипотезасы.

50 сұрақ, 5 түр × 10: фактілік, болжамдық, түсіндірмелі, what-if, құзыреттен тыс / ашық қате.
Әр сұрақтың эталоны сол сервистерді тікелей шақыру арқылы автоматты есептеледі.
Салыстырылады: «агент + құралдар + шлюз» және «құралсыз LLM» (бірдей GPT моделі).

Метрикалар:
  task success   — эталондық негізгі сандардың барлығы жауапта бар (±2% немесе дөңгелектеу дәлдігі),
                   түсіндірмелі сұрақтарда — негізгі фактор аталған; құзыреттен тыс — ашық бас тарту;
  faithfulness   — (18) формула: жауаптағы сандардың құрал/контекст деректерімен сәйкес үлесі;
  safe-failure   — ашық қате/бас тарту күтілетін сұрақтарда дұрыс бас тарту үлесі;
  құрал шақыру саны, жауап уақыты, токендер.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from app.agent.gate import extract_numbers, faithfulness
from app.core.config import get_settings
from app.core.farms import get_farm
from app.services import forecast_service as fs
from app.services import history_service as hs
from app.services.schedule_export import build_dayahead_schedule

REFUSAL = re.compile(
    r"алмаймын|мүмкін емес|қатысты емес|құзырет|жоқ|қолжетімсіз|ашық емес|кешіріңіз|ақпарат|деректер.*?(жоқ|ашық емес)|"
    r"не могу|недоступ|нет данных|вне|извините|не относится", re.I)
TYPES = {"factual": "Фактілік", "forecast": "Болжамдық", "explain": "Түсіндірмелі", "whatif": "What-if",
         "oos": "Құзыреттен тыс / ашық қате"}


@dataclass
class Q:
    id: str
    type: str
    farm: str
    text: str
    ref: Callable[[], dict] | None = None  # эталондық құрал нәтижесі
    keys: Callable[[dict], list] = field(default=lambda r: [])  # тексерілетін негізгі сандар
    keywords: tuple = ()  # түсіндірмелі сұрақтар үшін
    expect_refusal: bool = False


def _tomorrow(farm_id: str) -> tuple[str, str]:
    farm = get_farm(farm_id)
    d = (fs.to_local(fs.now_utc(), farm) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    nxt = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return d, nxt


def build_questions() -> list[Q]:
    tk, tk2 = _tomorrow("kelmarsh")
    te, te2 = _tomorrow("ereymentau")
    E = "energy_mwh_p50"
    pp = fs.predict_power
    qs: list[Q] = [
        # ---------------- фактілік (тарих және бэктест, Kelmarsh)
        Q("F01", "factual", "kelmarsh", "Kelmarsh станциясы 2019 жылғы наурызда қанша энергия өндірді?",
          lambda: hs.get_history("kelmarsh", "2019-03-01 00:00", "2019-04-01 00:00", False), lambda r: [r["summary"]["energy_mwh"]]),
        Q("F02", "factual", "kelmarsh", "2019 жылғы 1–7 шілде аралығында Kelmarsh-тың орташа қуаты қандай болды?",
          lambda: hs.get_history("kelmarsh", "2019-07-01 00:00", "2019-07-08 00:00", False), lambda r: [r["summary"]["mean_mw"]]),
        Q("F03", "factual", "kelmarsh", "Тест кезеңінде ансамбль моделінің 24 сағаттық горизонттағы nMAE мәні қандай?",
          lambda: hs.backtest("kelmarsh", "ensemble", horizon=24), lambda r: [r["metrics"]["nmae_pct"]]),
        Q("F04", "factual", "kelmarsh", "Persistence моделінің 6 сағаттық горизонттағы nMAE мәні қанша?",
          lambda: hs.backtest("kelmarsh", "persistence", horizon=6), lambda r: [r["metrics"]["nmae_pct"]]),
        Q("F05", "factual", "kelmarsh", "Тест кезеңінде 24 сағаттық горизонтта қай модель ең дәл болды және оның nMAE қанша?",
          lambda: hs.compare_models("kelmarsh", horizon=24), lambda r: [r["table"][0]["nmae_pct"]]),
        Q("F06", "factual", "kelmarsh", "LightGBM моделінің 48 сағаттық горизонттағы skill score мәні қандай?",
          lambda: hs.backtest("kelmarsh", "lgbm", horizon=48), lambda r: [r["metrics"]["skill_vs_persistence"]]),
        Q("F07", "factual", "kelmarsh", "2019 жылғы қаңтарда жел жеткілікті болса да турбиналар тоқтап тұрған кезеңдер болды ма? Қанша энергия жоғалды?",
          lambda: hs.detect_downtime("kelmarsh", "2019-01-01 00:00", "2019-02-01 00:00"), lambda r: [r["total_lost_energy_mwh"]] if r["n_periods"] else []),
        Q("F08", "factual", "kelmarsh", "Ансамбль моделінің 24 сағаттағы P10–P90 аралығын қамту үлесі (PICP) қандай?",
          lambda: hs.backtest("kelmarsh", "ensemble", horizon=24), lambda r: [r["metrics"]["picp_80_pct"]]),
        Q("F09", "factual", "kelmarsh", "2019 жылғы 15 қаңтарда Kelmarsh-тың максималды қуаты қанша болды?",
          lambda: hs.get_history("kelmarsh", "2019-01-15 00:00", "2019-01-16 00:00", False), lambda r: [r["summary"]["max_mw"]]),
        Q("F10", "factual", "kelmarsh", "Физикалық модельдің 24 сағаттық горизонттағы MAE мәні МВт-пен қандай?",
          lambda: hs.backtest("kelmarsh", "physical", horizon=24), lambda r: [r["metrics"]["mae_mw"]]),
        # ---------------- болжамдық (жедел)
        Q("P01", "forecast", "kelmarsh", "Ертең Kelmarsh тәулік бойы қанша энергия өндіреді?",
          lambda: pp("kelmarsh", 72, None, f"{tk} 00:00", f"{tk2} 00:00"), lambda r: [r["summary"][E]]),
        Q("P02", "forecast", "kelmarsh", "Ертең сағат 10:00-ден 18:00-ге дейін Kelmarsh қанша энергия өндіреді?",
          lambda: pp("kelmarsh", 72, None, f"{tk} 10:00", f"{tk} 18:00"), lambda r: [r["summary"][E]]),
        Q("P03", "forecast", "kelmarsh", "Келесі 24 сағатта Kelmarsh-тың максималды қуаты қашан және қанша болады?",
          lambda: pp("kelmarsh", 24), lambda r: [r["summary"]["max_mw_p50"]]),
        Q("P04", "forecast", "ereymentau", "Ерейментау ЖЭС ертең тәулік бойы қанша энергия өндіреді?",
          lambda: pp("ereymentau", 72, None, f"{te} 00:00", f"{te2} 00:00"), lambda r: [r["summary"][E]]),
        Q("P05", "forecast", "kelmarsh", "Келесі 72 сағатта Kelmarsh-тың қуатты пайдалану коэффициенті қандай болады?",
          lambda: pp("kelmarsh", 72), lambda r: [r["summary"]["capacity_factor_pct"]]),
        Q("P06", "forecast", "kelmarsh", "Kelmarsh үшін ертеңгі тәуліктік сағаттық кестені жасап, жалпы энергияны айт.",
          lambda: build_dayahead_schedule("kelmarsh", export=False), lambda r: [r["summary"][E]]),
        Q("P07", "forecast", "ereymentau", "Ерейментауда келесі 3 тәулікте ступица биіктігінде жел 4 м/с-тан төмен болатын, 6 сағаттан ұзақ кезеңдер бар ма?",
          lambda: fs.weather_forecast("ereymentau", 72, 4.0, 6), lambda r: [w["duration_h"] for w in r["low_wind_windows"]]),
        Q("P08", "forecast", "kelmarsh", "Ертең Kelmarsh-тың тәуліктік өндірісі P10 және P90 бойынша қандай аралықта болады?",
          lambda: pp("kelmarsh", 72, None, f"{tk} 00:00", f"{tk2} 00:00"), lambda r: [r["summary"]["energy_mwh_p10"], r["summary"]["energy_mwh_p90"]]),
        Q("P09", "forecast", "kelmarsh", "Келесі 48 сағаттағы ступица биіктігіндегі орташа жел жылдамдығы қандай?",
          lambda: fs.weather_forecast("kelmarsh", 48), lambda r: [r["summary"]["mean_ws_hub"]]),
        Q("P10", "forecast", "ereymentau", "Ерейментау бойынша келесі 72 сағатқа болжам есебін дайындап, жалпы энергияны айт.",
          lambda: pp("ereymentau", 72), lambda r: [r["summary"][E]]),
        # ---------------- түсіндірмелі
        *[Q(f"X{i:02d}", "explain", farm, text, lambda farm=farm, a=a, b=b: fs.explain_forecast(farm, a, b),
            lambda r: [], ("жел", "жылдамдығ"))
          for i, (farm, text, a, b) in enumerate([
              ("kelmarsh", "Ертеңгі болжамды түсіндір: өндіріске қандай факторлар көбірек әсер етеді?", f"{tk} 00:00", f"{tk2} 00:00"),
              ("kelmarsh", "Ертең кешке (18:00–23:00) Kelmarsh өндірісі неге өзгереді?", f"{tk} 18:00", f"{tk} 23:00"),
              ("ereymentau", "Ерейментаудың ертеңгі болжамына не әсер етеді?", f"{te} 00:00", f"{te2} 00:00"),
              ("kelmarsh", "Ертең таңертең (06:00–12:00) өндіріс неліктен осындай деңгейде?", f"{tk} 06:00", f"{tk} 12:00"),
              ("ereymentau", "Ерейментауда ертең түнгі (00:00–06:00) өндірісті не анықтайды?", f"{te} 00:00", f"{te} 06:00"),
              ("kelmarsh", "Келесі тәулікте ауа тығыздығы өндіріске әсер ете ме? Түсіндір.", None, None),
              ("kelmarsh", "Ертеңгі күндізгі (12:00–18:00) болжамның негізгі себептерін ата.", f"{tk} 12:00", f"{tk} 18:00"),
              ("ereymentau", "Ерейментауда ертең кешкі өндіріс (18:00–24:00) неге осылай болжанды?", f"{te} 18:00", f"{te2} 00:00"),
              ("kelmarsh", "Болжамға жел бағыты қаншалықты әсер етеді? Келесі 72 сағат бойынша түсіндір.", None, None),
              ("ereymentau", "Ерейментаудың келесі 72 сағаттағы болжамын түсіндір.", None, None),
          ], 1)],
        # ---------------- what-if
        *[Q(f"W{i:02d}", "whatif", farm, text, lambda farm=farm, d=d: fs.what_if(farm, d, 24),
            lambda r: [r["scenario"][E]])
          for i, (farm, d, text) in enumerate([
              ("kelmarsh", 15, "Егер жел болжамнан 15% күшті болса, келесі 24 сағатта Kelmarsh қанша энергия өндіреді?"),
              ("kelmarsh", -15, "Жел болжамнан 15% әлсіз болса, келесі 24 сағаттағы өндіріс қандай болады?"),
              ("kelmarsh", 10, "Жел 10%-ға күшті болса, келесі тәуліктегі энергия қанша болады?"),
              ("kelmarsh", -20, "Жел 20%-ға әлсіз болған жағдайда келесі 24 сағатта қанша өндіреміз?"),
              ("kelmarsh", 25, "Жел болжамнан 25% күшті болса, келесі 24 сағаттағы өндіріс қанша?"),
              ("ereymentau", 15, "Ерейментауда жел болжамнан 15% күшті болса, келесі 24 сағатта қанша энергия өндіріледі?"),
              ("ereymentau", -10, "Ерейментауда жел 10%-ға әлсіз болса, келесі тәуліктегі өндіріс қандай?"),
              ("ereymentau", 20, "Ерейментауда жел 20%-ға күшейсе, келесі 24 сағатта қанша өндіреміз?"),
              ("ereymentau", -25, "Ерейментауда жел 25%-ға әлсіресе, келесі 24 сағаттағы энергия қанша?"),
              ("kelmarsh", -30, "Жел 30%-ға әлсіз болса, келесі 24 сағатта Kelmarsh қанша өндіреді?"),
          ], 1)],
        # ---------------- құзыреттен тыс / ашық қате
        *[Q(f"O{i:02d}", "oos", farm, text, None, lambda r: [], (), True) for i, (farm, text) in enumerate([
            ("kelmarsh", "Python-да көпіршікті сұрыптау алгоритмін жазып бер."),
            ("kelmarsh", "Бүгін доллардың теңгеге бағамы қандай?"),
            ("ereymentau", "Ерейментау ЖЭС өткен айда нақты қанша энергия өндірді?"),
            ("kelmarsh", "Kelmarsh станциясы 2012 жылы қанша энергия өндірді?"),
            ("kelmarsh", "Ертең күн электр станциясы қанша өндіреді?"),
            ("kelmarsh", "Kelmarsh-тың 15 күннен кейінгі өндірісі қандай болады?"),
            ("kelmarsh", "Кеше болған футбол матчының есебі қандай?"),
            ("ereymentau", "Ерейментау үшін модельдің бэктест дәлдігі (nMAE) қандай?"),
            ("kelmarsh", "Маған жақсы мейрамхана ұсын."),
            ("ereymentau", "Ерейментаудағы турбиналардың өткен аптадағы тоқтап тұру кезеңдерін көрсет."),
        ], 1)],
    ]
    return qs


def _key_hit(answer: str, value: float) -> bool:
    if value is None:
        return False
    for tok, v in extract_numbers(answer):
        d = len(tok.split(",")[-1].split(".")[-1]) if re.search(r"[.,]\d+$", tok) else 0
        eps = max(0.5 * 10 ** (-d) + 1e-9, 0.02 * abs(value))
        if abs(v - value) <= eps or abs(abs(v) - abs(value)) <= eps:
            return True
    return False


def score(q: Q, answer: str, ref: dict | None, refs_for_faith: list) -> dict:
    nums = extract_numbers(answer)
    f = faithfulness(answer, refs_for_faith, q.text)
    refused = bool(REFUSAL.search(answer))
    if q.expect_refusal:
        success = refused and f["score"] >= 1.0
    elif q.type == "explain":
        success = any(k in answer.lower() for k in q.keywords) and f["score"] >= 1.0
    else:
        keys = q.keys(ref) if ref and "error" not in ref else []
        success = bool(keys) and all(_key_hit(answer, k) for k in keys)
        if not keys and ref is not None and not ref.get("error"):  # эталонда сан жоқ (мысалы, кезең табылмады)
            success = f["score"] >= 1.0 and (refused or not nums)
    return {"success": bool(success), "faithfulness": f["score"], "n_numbers": f["n_numbers"],
            "unsupported": f["unsupported"], "refused": refused}


def run(systems=("agent", "llm_only"), limit: int | None = None, log=print) -> pd.DataFrame:
    from app.agent import graph
    from app.agent.llm_only import ask_llm_only

    fs.WX_TTL_S = 3 * 3600  # бағалау кезінде ауа райы болжамы тұрақты болуы үшін
    qs = build_questions()[:limit] if limit else build_questions()
    out_dir = get_settings().reports_dir / "agent_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    for q in qs:
        try:
            ref = q.ref() if q.ref else None
        except Exception as e:
            ref = {"error": str(e)}
        for sysname in systems:
            try:
                if sysname == "agent":
                    r = graph.ask(q.text, "kk", q.farm, log=True)
                    faith_refs = [ref] if ref else []
                    faith_refs += [a for a in r.get("artifacts", [])] + [t for t in r.get("trace", []) if t.get("step") == "tool"]
                    n_tools = len(r.get("tools_called", []))
                else:
                    r = ask_llm_only(q.text, q.farm, "kk")
                    faith_refs = ([ref] if ref else []) + [r["context_data"]]
                    n_tools = 0
                sc = score(q, r["answer"], ref, faith_refs)
            except Exception as e:
                r, sc, n_tools = {"answer": f"ERROR: {e}", "latency_s": None, "tokens": None}, \
                    {"success": False, "faithfulness": None, "n_numbers": 0, "unsupported": [], "refused": False}, 0
            rows.append({"id": q.id, "type": q.type, "farm": q.farm, "system": sysname, "question": q.text,
                         "answer": r["answer"], "latency_s": r.get("latency_s"), "tokens": r.get("tokens"),
                         "tool_calls": n_tools, "gate_status": r.get("status"), **sc,
                         "reference_keys": json.dumps(q.keys(ref) if ref and "error" not in ref else [], default=float)})
            if sysname == "agent" and q.id in ("P02", "X02", "W01", "O03"):
                examples.append({"id": q.id, "question": q.text, "answer": r["answer"], "trace": r.get("trace"),
                                 "gate": r.get("gate")})
            log(f"  {q.id} [{sysname}] success={sc['success']} faith={sc['faithfulness']}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    (out_dir / "trajectory_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summarize(df)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sysname, g in df.groupby("system"):
        oos = g[g.type == "oos"]
        rows.append({
            "жүйе": "ЖИ-агент (құралдар + шлюз)" if sysname == "agent" else "Құралсыз LLM",
            "тапсырманы орындау, %": round(100 * g.success.mean(), 1),
            "faithfulness": round(float(g.faithfulness.dropna().mean()), 3),
            "ойдан шығарылған сандар": int(g.unsupported.apply(lambda u: len(u) if isinstance(u, list) else 0).sum()),
            "қатені ашық хабарлау үлесі, %": round(100 * oos.success.mean(), 1) if len(oos) else None,
            "құрал шақыру саны (орт.)": round(float(g.tool_calls.mean()), 2),
            "жауап уақыты, с (орт.)": round(float(g.latency_s.dropna().mean()), 2),
            "токендер (орт.)": int(g.tokens.dropna().mean()) if g.tokens.notna().any() else None,
        })
    t = pd.DataFrame(rows)
    by_type = df.pivot_table(index="type", columns="system", values="success", aggfunc="mean").mul(100).round(1)
    d = get_settings().reports_dir / "tables"
    t.to_csv(d / "tab17_agent_vs_llm.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(d / "tab17b_success_by_type.csv", encoding="utf-8-sig")
    (d / "tab17_agent_vs_llm.md").write_text(
        "**Кесте 17 — ЖИ-агент пен құралсыз LLM-ді салыстыру нәтижелері**\n\n" + t.to_string(index=False)
        + "\n\nСұрақ түрлері бойынша тапсырманы орындау, %:\n\n" + by_type.to_string() + "\n", encoding="utf-8")
    return t


def export_questions() -> None:
    """Қосымша 3: тест сұрақтарының тізімі (эталондық жауаптары)."""
    qs = build_questions()
    rows = []
    for q in qs:
        try:
            ref = q.ref() if q.ref else None
            keys = q.keys(ref) if ref else []
        except Exception as e:
            ref, keys = {"error": str(e)}, []
        rows.append({"id": q.id, "түрі": TYPES[q.type], "станция": q.farm, "сұрақ": q.text,
                     "эталон (негізгі сандар)": ", ".join(f"{k:g}" if isinstance(k, (int, float, np.floating)) else str(k) for k in keys)
                     if keys else ("ашық бас тарту / қате хабарламасы" if q.expect_refusal else
                                   "негізгі фактор: жел жылдамдығы (SHAP)" if q.type == "explain" else "—")})
    d = get_settings().reports_dir / "agent_eval"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "questions_appendix3.csv", index=False, encoding="utf-8-sig")
