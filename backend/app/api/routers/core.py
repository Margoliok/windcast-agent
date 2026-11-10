"""REST API: станциялар, болжам, тарих, метрикалар, сценарий, тоқтап тұру, кесте, есеп."""
from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.farms import FARMS, get_farm
from app.models.registry import bundle_path, load_bundle
from app.services import forecast_service as fs
from app.services import history_service as hs
from app.services.report import generate_report
from app.services.schedule_export import build_dayahead_schedule

router = APIRouter(prefix="/api")


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except (fs.ServiceError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "llm_configured": bool(s.openai_api_key), "llm_model": s.openai_model,
            "models_trained": {f: bundle_path(f).exists() for f in FARMS}}


@router.get("/farms")
def farms():
    out = []
    for f in FARMS.values():
        d = f.to_dict()
        if bundle_path(f.id).exists():
            b = load_bundle(f.id)
            d["models"] = b.available_models
            d["best_model"] = b.best_model
            d["data_mode"] = b.meta.get("mode")
        out.append(d)
    return out


@router.get("/forecast")
def forecast(farm_id: str = "kelmarsh", hours: int = Query(72, ge=1, le=168), model: str | None = None,
             start_local: str | None = None, end_local: str | None = None):
    return _call(fs.predict_power, farm_id, hours, model, start_local, end_local)


@router.get("/weather")
def weather(farm_id: str = "kelmarsh", hours: int = Query(72, ge=1, le=168), low_wind_threshold: float | None = None,
            min_duration_h: int = 6):
    return _call(fs.weather_forecast, farm_id, hours, low_wind_threshold, min_duration_h)


@router.get("/explain")
def explain(farm_id: str = "kelmarsh", start_local: str | None = None, end_local: str | None = None, lang: str = "kk"):
    return _call(fs.explain_forecast, farm_id, start_local, end_local, lang=lang)


@router.get("/whatif")
def whatif(farm_id: str = "kelmarsh", delta_wind_pct: float = 15, hours: int = 24):
    return _call(fs.what_if, farm_id, delta_wind_pct, hours)


@router.get("/history")
def history(farm_id: str = "kelmarsh", start_local: str | None = None, end_local: str | None = None):
    return _call(hs.get_history, farm_id, start_local, end_local)


@router.get("/downtime")
def downtime(farm_id: str = "kelmarsh", start_local: str | None = None, end_local: str | None = None,
             min_duration_h: int = 2):
    return _call(hs.detect_downtime, farm_id, start_local, end_local, min_duration_h)


@router.get("/models/compare")
def compare(farm_id: str = "kelmarsh", horizon: int = 24, start_local: str | None = None, end_local: str | None = None):
    return _call(hs.compare_models, farm_id, start_local, end_local, horizon)


@router.get("/models/backtest")
def backtest(farm_id: str = "kelmarsh", model: str = "ensemble", horizon: int = 24, start_local: str | None = None,
             end_local: str | None = None):
    return _call(hs.backtest, farm_id, model, start_local, end_local, horizon)


@router.get("/models/backtest_series")
def backtest_series(farm_id: str = "kelmarsh", model: str = "ensemble", horizon: int = 24,
                    start_local: str | None = None, end_local: str | None = None):
    return _call(hs.backtest_series, farm_id, model, horizon, start_local, end_local)


@router.get("/models/horizons")
def horizons():
    """Сурет 16 деректері: nMAE(h) барлық модельдер бойынша."""
    p = get_settings().reports_dir / "tables" / "fig16_nmae_by_horizon.csv"
    if not p.exists():
        raise HTTPException(503, "Эксперименттер әлі іске қосылмаған")
    return pd.read_csv(p).to_dict(orient="records")


@router.get("/models/info")
def models_info(farm_id: str = "kelmarsh"):
    return _call(fs.model_list, farm_id)


@router.get("/experiments/summary")
def experiments_summary():
    p = get_settings().reports_dir / "tables" / "experiments_summary.json"
    if not p.exists():
        raise HTTPException(503, "Эксперименттер әлі іске қосылмаған")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/experiments/table/{name}")
def experiment_table(name: str):
    p = get_settings().reports_dir / "tables" / f"{name}.csv"
    if not p.exists() or "/" in name or ".." in name:
        raise HTTPException(404, "Кесте табылмады")
    return pd.read_csv(p).to_dict(orient="records")


@router.get("/schedule")
def schedule(farm_id: str = "kelmarsh", date_local: str | None = None, model: str | None = None, lang: str = "kk"):
    return _call(build_dayahead_schedule, farm_id, date_local, model, lang)


@router.post("/reports")
def report(farm_id: str = "kelmarsh", hours: int = 72, lang: str = "kk"):
    return _call(generate_report, farm_id, hours, None, lang)


@router.get("/exports/{fname}")
def exports(fname: str):
    d = get_settings().exports_dir.resolve()
    p = (d / fname).resolve()
    if p.parent != d or not p.exists():
        raise HTTPException(404, "Файл табылмады")
    return FileResponse(p, filename=fname)


@router.get("/farms/{farm_id}/curve")
def curve(farm_id: str):
    get_farm(farm_id)
    b = _call(load_bundle, farm_id)
    return {"speeds": b.curve.speeds.tolist(), "power_norm": b.curve.power.tolist(), "source": b.curve.source,
            "capacity_mw": get_farm(farm_id).capacity_mw}


@router.get("/forecast/latest")
def forecast_latest(farm_id: str = "kelmarsh"):
    """Жоспарлаушы дерекқорға жазған соңғы болжам (forecasts кестесі)."""
    from sqlalchemy import func, select

    from app.core.db import session
    from app.db.models import ForecastRow

    farm = get_farm(farm_id)
    with session() as s:
        last = s.execute(select(func.max(ForecastRow.issue_time)).where(ForecastRow.farm_id == farm_id)).scalar()
        if last is None:
            raise HTTPException(404, "Дерекқорда әлі болжам жоқ")
        rows = s.execute(select(ForecastRow).where(ForecastRow.farm_id == farm_id, ForecastRow.issue_time == last)
                         .order_by(ForecastRow.valid_time)).scalars().all()
        return {"farm_id": farm_id, "issue_time_local": fs.fmt_local(pd.Timestamp(last), farm),
                "model_version": rows[0].model_version if rows else None,
                "series": [{"time_local": fs.fmt_local(pd.Timestamp(r.valid_time), farm), "p10": r.p10, "p50": r.p50,
                            "p90": r.p90, "gate_passed": r.gate_passed} for r in rows]}
