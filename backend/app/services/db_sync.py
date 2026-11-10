"""Өңделген деректерді дерекқорға жүктеу: станциялар, SCADA (сағаттық), ERA5, бэктест метрикалары."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, insert, select

from app.core.config import get_settings
from app.core.db import init_db, session
from app.core.farms import FARMS
from app.db.models import FarmRow, MetricRow, ScadaMeasurement, WeatherHistory


def _nz(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def seed_farms() -> None:
    from app.models.registry import bundle_path, load_bundle

    with session() as s:
        for f in FARMS.values():
            curve = None
            if bundle_path(f.id).exists():
                c = load_bundle(f.id).curve
                curve = {"speeds": c.speeds.tolist(), "power": c.power.tolist(), "source": c.source}
            s.merge(FarmRow(id=f.id, name=f.name, lat=f.lat, lon=f.lon, capacity_mw=f.capacity_mw,
                            hub_height_m=f.hub_height_m, n_turbines=f.n_turbines, turbine_model=f.turbine_model,
                            timezone=f.timezone, power_curve=curve))


def sync_history(force: bool = False, log=print) -> dict:
    p = get_settings().processed_dir
    if not (p / "kelmarsh_power.parquet").exists():
        log("  өңделген деректер жоқ — sync өткізілді")
        return {"scada": 0, "weather": 0}
    with session() as s:
        n = s.execute(select(func.count()).select_from(ScadaMeasurement)).scalar()
    if n and not force:
        return {"scada": int(n), "weather": None, "skipped": True}
    power = pd.read_parquet(p / "kelmarsh_power.parquet")
    era5 = pd.read_parquet(p / "kelmarsh_wx_era5.parquet").reindex(power.index)
    outage = power.get("f_outage", pd.Series(False, index=power.index)).fillna(False)
    scada_rows = [{
        "farm_id": "kelmarsh", "timestamp": ts.to_pydatetime(),
        "active_power_mw": _nz(r.power_mw), "raw_power_mw": _nz(r.power_raw_mw),
        "availability": _nz(r.availability) if "availability" in power else None,
        "wind_speed": _nz(r.nacelle_ws) if "nacelle_ws" in power else None,
        "status": "outage" if outage.get(ts, False) else ("missing" if pd.isna(r.power_raw_mw) else "ok"),
    } for ts, r in power.iterrows()]
    wx_rows = [{
        "farm_id": "kelmarsh", "timestamp": ts.to_pydatetime(), "wind_speed_10m": _nz(r.ws10),
        "wind_speed_100m": _nz(r.ws100), "wind_direction": _nz(r.wd100), "temperature": _nz(r.t2m), "pressure": _nz(r.sp),
    } for ts, r in era5.iterrows()]
    with session() as s:
        s.execute(delete(ScadaMeasurement).where(ScadaMeasurement.farm_id == "kelmarsh"))
        s.execute(delete(WeatherHistory).where(WeatherHistory.farm_id == "kelmarsh"))
        for i in range(0, len(scada_rows), 5000):
            s.execute(insert(ScadaMeasurement), scada_rows[i:i + 5000])
        for i in range(0, len(wx_rows), 5000):
            s.execute(insert(WeatherHistory), wx_rows[i:i + 5000])
    log(f"  DB: scada={len(scada_rows)}, weather_history={len(wx_rows)}")
    return {"scada": len(scada_rows), "weather": len(wx_rows)}


def sync_metrics(log=print) -> int:
    f = get_settings().reports_dir / "tables" / "e1_metrics_full.csv"
    if not f.exists():
        return 0
    t = pd.read_csv(f)
    meta_p = get_settings().processed_dir / "dataset_meta.json"
    period = "test"
    if meta_p.exists():
        import json

        sp = json.loads(meta_p.read_text(encoding="utf-8"))["splits"]["test"]
        period = f"{sp[0][:10]}..{sp[1][:10]}"
    with session() as s:
        s.execute(delete(MetricRow).where(MetricRow.farm_id == "kelmarsh", MetricRow.period == period))
        s.execute(insert(MetricRow), [{
            "farm_id": "kelmarsh", "model_version": r.model, "period": period, "horizon": int(r.h),
            "mae": float(r.mae), "rmse": float(r.rmse), "nmae": float(r.nmae),
            "crps": _nz(r.crps), "picp": _nz(r.picp_80)} for r in t.itertuples()])
    log(f"  DB: metrics={len(t)}")
    return len(t)


def sync_all(force: bool = False, log=print) -> None:
    init_db()
    seed_farms()
    sync_history(force=force, log=log)
    sync_metrics(log=log)
