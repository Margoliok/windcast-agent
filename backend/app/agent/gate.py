"""Тексеру шлюзі (verification gate) — код, LLM емес.

1. Физикалық шектеулер: 0 ≤ P ≤ P_inst, P10 ≤ P50 ≤ P90, NaN жоқ, ұзындық күтілгенге тең,
   өлшем бірлігі — МВт.
2. Faithfulness (18): жауаптағы әр сан құралдардың нәтижесіндегі қандай да бір санмен
   ε дәлдігімен сәйкес келуі керек:  Faith = (1/K) Σ 1{|r_k − z_k| ≤ ε_k}.
   Сұрақтағы сандар, күн/уақыт белгілері және тізім нөмірлері есепке алынбайды.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

TOL_ABS = 1e-6


@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- physical checks
def check_series(rows: list[dict], capacity_mw: float, expected_len: int | None = None, unit: str = "MW") -> GateResult:
    checks = []

    def add(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    p10 = np.array([r.get("p10", np.nan) for r in rows], dtype=float)
    p50 = np.array([r.get("p50", np.nan) for r in rows], dtype=float)
    p90 = np.array([r.get("p90", np.nan) for r in rows], dtype=float)
    add("unit", unit == "MW", f"unit={unit}")
    add("no_nan", bool(len(rows)) and np.isfinite(p10).all() and np.isfinite(p50).all() and np.isfinite(p90).all(),
        f"n={len(rows)}")
    eps = 1e-6
    allv = np.concatenate([p10, p50, p90])
    allv = allv[np.isfinite(allv)]
    lo = float(allv.min()) if allv.size else 0.0
    hi = float(allv.max()) if allv.size else 0.0
    add("bounds", lo >= -eps and hi <= capacity_mw + eps, f"min={lo:.3f}, max={hi:.3f}, P_inst={capacity_mw}")
    order_ok = bool(np.all(p10 <= p50 + eps) and np.all(p50 <= p90 + eps))
    add("quantile_order", order_ok, "P10 ≤ P50 ≤ P90")
    if expected_len is not None:
        add("length", len(rows) == expected_len, f"{len(rows)}/{expected_len}")
    return GateResult(all(c["passed"] for c in checks), checks)


# --------------------------------------------------------------------------- faithfulness
_NUM = re.compile(r"(?<![\w.])[-−]?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?!\d)|(?<![\w.])[-−]?\d+(?:[.,]\d+)?(?![\d])")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_LIST = re.compile(r"(?m)^\s*\d+[.)]\s")
_PNN = re.compile(r"\b[PpРр]\s?(10|50|90)\b|\bq0\.\d\b|\b[HhМм]\d\b|\bM\d\b|\bМ\d\b")
_HEADING = re.compile(r"(?m)^#+.*$")


def _to_float(tok: str) -> float | None:
    t = tok.replace("−", "-").replace(" ", "").replace(" ", "").replace(" ", "")
    if t.count(",") == 1 and "." not in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _decimals(tok: str) -> int:
    m = re.search(r"[.,](\d+)$", tok.strip())
    return len(m.group(1)) if m else 0


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """Мәтіндегі «тексерілетін» сандар (күн, уақыт, тізім нөмірі, P10/P50/P90 белгілері алынып тасталады)."""
    clean = _DATE.sub(" ", text)
    clean = _TIME.sub(" ", clean)
    clean = _LIST.sub(" ", clean)
    clean = _PNN.sub(" ", clean)
    out = []
    for m in _NUM.finditer(clean):
        tok = m.group(0)
        v = _to_float(tok)
        if v is not None:
            out.append((tok.strip(), v))
    return out


def collect_numbers(obj: Any, acc: list[float] | None = None) -> list[float]:
    """Құрал нәтижесіндегі барлық сандарды рекурсивті жинау (мәтін ішіндегілерін де)."""
    acc = [] if acc is None else acc
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float, np.integer, np.floating)):
        if math.isfinite(float(obj)):
            acc.append(float(obj))
    elif isinstance(obj, str):
        for _, v in extract_numbers(obj):
            acc.append(v)
        for t in _TIME.findall(obj):  # «14:00» -> 14
            acc.append(float(t.split(":")[0]))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, acc)
    return acc


def _matches(value: float, tok: str, refs: np.ndarray) -> bool:
    if refs.size == 0:
        return False
    d = _decimals(tok)
    eps = max(0.5 * 10 ** (-d) + 1e-9, 0.01 * abs(value))  # дөңгелектеу немесе 1% салыстырмалы
    cands = [refs, refs * 1000.0, refs / 1000.0, refs * 100.0]  # МВт↔кВт, МВт·сағ↔ГВт·сағ, үлес↔%
    return any(np.any(np.abs(c - value) <= eps) or np.any(np.abs(np.abs(c) - abs(value)) <= eps) for c in cands)


def faithfulness(answer: str, tool_outputs: Iterable[Any], question: str = "") -> dict:
    refs = np.array(collect_numbers(list(tool_outputs)), dtype=float)
    q_nums = {round(v, 6) for _, v in extract_numbers(question)}
    items = []
    for tok, v in extract_numbers(answer):
        if round(v, 6) in q_nums:
            continue  # пайдаланушы өзі айтқан сан
        if abs(v) < 1 and _decimals(tok) == 0:
            continue  # «0» сияқты тривиал сандар
        items.append({"value": tok, "supported": _matches(v, tok, refs)})
    k = len(items)
    score = 1.0 if k == 0 else sum(i["supported"] for i in items) / k
    return {
        "score": score,
        "n_numbers": k,
        "unsupported": [i["value"] for i in items if not i["supported"]],
        "items": items,
    }


def verify_answer(answer: str, tool_outputs: list[Any], question: str = "", threshold: float = 1.0) -> GateResult:
    """Агенттің соңғы жауабын тексеру: құрал нәтижелерінің физикалық тексерісі + faithfulness."""
    checks = []
    for out in tool_outputs:
        if isinstance(out, dict) and isinstance(out.get("gate"), dict) and not out["gate"].get("passed", True):
            checks.append({"check": f"tool_gate:{out.get('tool', '?')}", "passed": False,
                           "detail": "құрал нәтижесі физикалық тексеруден өтпеді"})
    f = faithfulness(answer, tool_outputs, question)
    checks.append({"check": "faithfulness", "passed": f["score"] >= threshold,
                   "detail": f"score={f['score']:.2f}, unsupported={f['unsupported']}", "score": f["score"],
                   "unsupported": f["unsupported"], "n_numbers": f["n_numbers"]})
    return GateResult(all(c["passed"] for c in checks), checks)
