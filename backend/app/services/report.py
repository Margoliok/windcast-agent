"""Графиктері бар есеп (Markdown + PDF): болжам, әлсіз жел кезеңдері, модельдер сапасы."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.farms import get_farm  # noqa: E402
from app.services.forecast_service import ServiceError, forecast_payload, run_forecast, weather_forecast  # noqa: E402

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
T = {
    "kk": {"title": "WindCast Agent — болжам есебі", "farm": "Станция", "issued": "Шығарылған уақыты",
           "model": "Модель", "summary": "Қорытынды", "energy": "Энергия (P50)", "range": "Аралық (P10–P90)",
           "mean": "Орташа қуат", "max": "Максимум", "min": "Минимум", "cf": "Қуатты пайдалану коэффициенті",
           "low": "Әлсіз жел кезеңдері (ступица < 4 м/с, ≥ 6 сағ)", "none": "жоқ", "quality": "Модельдер сапасы (тест кезеңі, 24 сағ)",
           "gate": "Тексеру шлюзі", "passed": "өтті", "failed": "өтпеді", "hours": "сағ"},
    "ru": {"title": "WindCast Agent — отчёт о прогнозе", "farm": "Станция", "issued": "Время выпуска",
           "model": "Модель", "summary": "Итоги", "energy": "Энергия (P50)", "range": "Интервал (P10–P90)",
           "mean": "Средняя мощность", "max": "Максимум", "min": "Минимум", "cf": "Коэффициент использования мощности",
           "low": "Периоды слабого ветра (ступица < 4 м/с, ≥ 6 ч)", "none": "нет", "quality": "Качество моделей (тестовый период, 24 ч)",
           "gate": "Верификационный шлюз", "passed": "пройден", "failed": "не пройден", "hours": "ч"},
}


def _font() -> str | None:
    return next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


def _chart(payload: dict, path: Path) -> None:
    s = pd.DataFrame(payload["series"])
    t = pd.to_datetime(s.time_local)
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=130)
    ax.fill_between(t, s.p10, s.p90, color="#7fb8d4", alpha=0.45, label="P10–P90")
    ax.plot(t, s.p50, color="#0b5e7e", lw=2, label="P50")
    ax.axhline(payload["capacity_mw"], color="#999", ls="--", lw=0.8)
    ax.set_ylabel("MW")
    ax.set_ylim(0, payload["capacity_mw"] * 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_report(farm_id: str, hours: int = 72, model: str | None = None, lang: str = "kk") -> dict:
    lang = lang if lang in T else "kk"
    tr = T[lang]
    farm = get_farm(farm_id)
    fc = run_forecast(farm_id, hours, model)
    payload = forecast_payload(fc)
    wx = weather_forecast(farm_id, hours, low_wind_threshold=4.0, min_duration_h=6)
    quality = None
    if farm.has_scada:
        try:
            from app.services.history_service import compare_models

            quality = compare_models(farm_id, horizon=24)
        except ServiceError:
            quality = None
    sm = payload["summary"]
    d = get_settings().exports_dir
    stem = f"report_{farm_id}_{fc.issue_time:%Y%m%d_%H%M}"
    png = d / f"{stem}.png"
    _chart(payload, png)

    lines = [
        f"# {tr['title']}", "",
        f"- {tr['farm']}: **{farm.name}** ({farm.capacity_mw} MW, {farm.timezone})",
        f"- {tr['issued']}: {payload['issue_time_local']}",
        f"- {tr['model']}: {payload['model']}",
        f"- {tr['gate']}: {tr['passed'] if payload['gate']['passed'] else tr['failed']}", "",
        f"## {tr['summary']} ({sm['period_local']})", "",
        f"- {tr['energy']}: **{sm['energy_mwh_p50']} MWh**",
        f"- {tr['range']}: {sm['energy_mwh_p10']} – {sm['energy_mwh_p90']} MWh",
        f"- {tr['mean']}: {sm['mean_mw_p50']} MW; {tr['cf']}: {sm['capacity_factor_pct']}%",
        f"- {tr['max']}: {sm['max_mw_p50']} MW ({sm['max_time_local']}); {tr['min']}: {sm['min_mw_p50']} MW ({sm['min_time_local']})",
        "", f"![forecast]({png.name})", "", f"## {tr['low']}", "",
    ]
    lw = wx.get("low_wind_windows", [])
    lines += [f"- {w['start_local']} – {w['end_local']} ({w['duration_h']} {tr['hours']}, {w['mean_ws_hub']} m/s)" for w in lw] or [f"- {tr['none']}"]
    if quality:
        lines += ["", f"## {tr['quality']}", "", "| Model | nMAE, % | Skill | CRPS, % | PICP80, % |", "|---|---|---|---|---|"]
        lines += [f"| {r['model_name']} | {r['nmae_pct']} | {r['skill_vs_persistence']} | {r['crps_pct']} | {r['picp_80_pct']} |"
                  for r in quality["table"]]
    md = "\n".join(lines)
    md_path = d / f"{stem}.md"
    md_path.write_text(md, encoding="utf-8")

    pdf_url = None
    font = _font()
    if font:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("U", "", font)
        for ln in lines:
            if ln.startswith("!["):
                pdf.image(str(png), w=185)
                continue
            if ln.startswith("|---"):
                continue
            size = 16 if ln.startswith("# ") else 13 if ln.startswith("## ") else 10
            pdf.set_font("U", size=size)
            text = ln.lstrip("#").strip().replace("**", "")
            if ln.startswith("|"):
                text = "   ".join(c.strip() for c in ln.strip("|").split("|"))
            pdf.multi_cell(0, 6 if size == 10 else 9, text or " ", new_x="LMARGIN", new_y="NEXT")
        pdf_path = d / f"{stem}.pdf"
        pdf.output(str(pdf_path))
        pdf_url = f"/api/exports/{pdf_path.name}"
    return {
        "farm": farm.name, "period_local": sm["period_local"], "summary": sm,
        "low_wind_windows": lw, "model_quality": quality["table"] if quality else None,
        "gate": payload["gate"],
        "files": {"markdown": f"/api/exports/{md_path.name}", "pdf": pdf_url, "chart": f"/api/exports/{png.name}"},
    }
