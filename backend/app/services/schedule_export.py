"""Келесі тәулікке сағаттық өндіріс кестесі (Кесте 18) және оны CSV/XLSX-ке экспорттау."""
from __future__ import annotations

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.agent.gate import check_series
from app.core.config import get_settings
from app.core.farms import get_farm
from app.services.forecast_service import ServiceError, fmt_local, now_utc, parse_local, run_forecast, summarize, to_local

HEADERS = {
    "kk": ["Сағат (жергілікті)", "P10, МВт", "P50, МВт", "P90, МВт", "Жел (ступица), м/с"],
    "ru": ["Час (местное)", "P10, МВт", "P50, МВт", "P90, МВт", "Ветер (ступица), м/с"],
}


def build_dayahead_schedule(farm_id: str, date_local: str | None = None, model: str | None = None,
                            lang: str = "kk", export: bool = True) -> dict:
    farm = get_farm(farm_id)
    now_loc = to_local(now_utc(), farm)
    day = pd.Timestamp(date_local).date() if date_local else (now_loc + pd.Timedelta(days=1)).date()
    start_utc = parse_local(f"{day} 00:00", farm)
    end_utc = start_utc + pd.Timedelta(hours=24)
    hours_needed = int((end_utc - now_utc()) / pd.Timedelta(hours=1))
    if hours_needed <= 0:
        raise ServiceError("Тәуліктік кесте тек болашақ күндер үшін жасалады.")
    if hours_needed > 168:
        raise ServiceError("Болжам горизонты 7 тәуліктен аспайды.")
    fc = run_forecast(farm_id, max(hours_needed, 24), model)
    f = fc.frame[(fc.frame.index >= start_utc) & (fc.frame.index < end_utc)]
    if len(f) < 24:
        raise ServiceError("Ауа райы болжамы бұл тәулікті толық қамтымайды.")
    rows = []
    for ts, r in f.iterrows():
        rows.append({"hour_local": to_local(ts, farm).strftime("%Y-%m-%d %H:%M"),
                     "p10": round(float(r.p10), 3), "p50": round(float(r.p50), 3), "p90": round(float(r.p90), 3),
                     "ws_hub": round(float(r.ws_hub), 2)})
    gate = check_series(rows, farm.capacity_mw, expected_len=24)
    out = {
        "farm": farm.name, "date_local": str(day), "timezone": farm.timezone, "model": fc.model, "unit": "MW",
        "issue_time_local": fmt_local(fc.issue_time, farm),
        "summary": summarize(fc, start_utc, end_utc), "schedule": rows, "gate": gate.to_dict(),
    }
    if export and gate.passed:
        out["files"] = export_files(out, lang)
    return out


def export_files(sched: dict, lang: str = "kk") -> dict:
    d = get_settings().exports_dir
    d.mkdir(parents=True, exist_ok=True)
    stem = f"schedule_{sched['farm'].split()[0].lower()}_{sched['date_local']}".replace(" ", "_")
    stem = stem.encode("ascii", "ignore").decode() or "schedule"
    df = pd.DataFrame(sched["schedule"])
    df.columns = HEADERS.get(lang, HEADERS["kk"])
    csv = d / f"{stem}.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    xlsx = d / f"{stem}.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Schedule"
    title = {"kk": "Тәуліктік өндіріс кестесі", "ru": "Суточный график выработки"}[lang if lang in ("kk", "ru") else "kk"]
    ws["A1"] = f"{title}: {sched['farm']}, {sched['date_local']} ({sched['timezone']})"
    ws["A1"].font = Font(bold=True, size=12)
    ws["A2"] = f"Model: {sched['model']} · issue: {sched['issue_time_local']} · WindCast Agent"
    for j, h in enumerate(df.columns, 1):
        c = ws.cell(row=4, column=j, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F6F8B")
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(j)].width = 20 if j == 1 else 14
    for i, row in enumerate(df.itertuples(index=False), 5):
        for j, v in enumerate(row, 1):
            ws.cell(row=i, column=j, value=v)
    last = 4 + len(df)
    ws.cell(row=last + 1, column=1, value="Σ, МВт·сағ").font = Font(bold=True)
    for j, col in ((2, "B"), (3, "C"), (4, "D")):
        ws.cell(row=last + 1, column=j, value=f"=SUM({col}5:{col}{last})").font = Font(bold=True)
    wb.save(xlsx)
    return {"csv": f"/api/exports/{csv.name}", "xlsx": f"/api/exports/{xlsx.name}"}
