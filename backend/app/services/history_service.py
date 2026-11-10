"""Тарихи деректер, бэктест, модельдерді салыстыру, тоқтап тұру кезеңдерін анықтау."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.db import session
from app.core.farms import get_farm
from app.db.models import ScadaMeasurement, WeatherHistory
from app.evaluation.backtest import metrics_table
from app.evaluation.experiments import MODEL_NAMES
from app.services.forecast_service import ServiceError, fmt_local, parse_local


def _require_scada(farm_id: str):
    farm = get_farm(farm_id)
    if not farm.has_scada:
        raise ServiceError(
            f"{farm.name} станциясының SCADA (нақты өндіріс) деректері ашық емес. "
            "Тарихи талдау мен бэктест тек Kelmarsh станциясы үшін қолжетімді."
        )
    return farm


def data_range(farm_id: str) -> dict:
    farm = _require_scada(farm_id)
    with session() as s:
        a, b = s.execute(select(func.min(ScadaMeasurement.timestamp), func.max(ScadaMeasurement.timestamp))
                         .where(ScadaMeasurement.farm_id == farm_id)).one()
    if a is None:
        raise ServiceError("Дерекқорда тарихи деректер жоқ. scripts/02_build_dataset.py және db sync іске қосыңыз.")
    return {"start_local": fmt_local(pd.Timestamp(a), farm), "end_local": fmt_local(pd.Timestamp(b), farm)}


def _load_history(farm_id: str, s_utc: pd.Timestamp, e_utc: pd.Timestamp) -> pd.DataFrame:
    with session() as s:
        rows = s.execute(
            select(ScadaMeasurement.timestamp, ScadaMeasurement.active_power_mw, ScadaMeasurement.raw_power_mw,
                   ScadaMeasurement.status, WeatherHistory.wind_speed_100m)
            .join(WeatherHistory, (WeatherHistory.farm_id == ScadaMeasurement.farm_id)
                  & (WeatherHistory.timestamp == ScadaMeasurement.timestamp), isouter=True)
            .where(ScadaMeasurement.farm_id == farm_id, ScadaMeasurement.timestamp >= s_utc,
                   ScadaMeasurement.timestamp < e_utc)
            .order_by(ScadaMeasurement.timestamp)
        ).all()
    return pd.DataFrame(rows, columns=["time", "power_mw", "raw_power_mw", "status", "ws100_era5"]).set_index("time")


def _period(farm, start_local, end_local, default_days=7):
    rng = data_range(farm.id)
    end_default = parse_local(rng["end_local"], farm) + pd.Timedelta(hours=1)
    e = parse_local(end_local, farm, end_default)
    s = parse_local(start_local, farm, e - pd.Timedelta(days=default_days))
    if s >= e:
        raise ServiceError("Кезеңнің басы соңынан ерте болуы керек.")
    return s, e, rng


def get_history(farm_id: str, start_local: str | None = None, end_local: str | None = None,
                include_series: bool = True) -> dict:
    farm = _require_scada(farm_id)
    s, e, rng = _period(farm, start_local, end_local)
    df = _load_history(farm_id, s, e)
    if df.empty:
        raise ServiceError(f"Бұл кезеңге деректер жоқ. Қолжетімді кезең: {rng['start_local']} – {rng['end_local']}.")
    valid = df.power_mw.dropna()
    out = {
        "farm": farm.name, "period_local": f"{fmt_local(df.index[0], farm)} – {fmt_local(df.index[-1] + pd.Timedelta(hours=1), farm)}",
        "available_range": rng, "unit": "MW",
        "summary": {
            "hours": int(len(df)), "valid_hours": int(len(valid)),
            "energy_mwh": round(float(df.raw_power_mw.clip(lower=0).sum()), 1),
            "mean_mw": round(float(valid.mean()), 2) if len(valid) else None,
            "max_mw": round(float(valid.max()), 2) if len(valid) else None,
            "capacity_factor_pct": round(float(df.raw_power_mw.clip(lower=0).mean() / farm.capacity_mw * 100), 1),
            "mean_ws100_era5": round(float(df.ws100_era5.mean()), 2),
            "outage_hours": int((df.status == "outage").sum()),
        },
    }
    if include_series:
        out["series"] = [{"time_local": fmt_local(t, farm), "power_mw": None if pd.isna(r.raw_power_mw) else round(float(r.raw_power_mw), 3),
                          "clean_power_mw": None if pd.isna(r.power_mw) else round(float(r.power_mw), 3),
                          "ws100_era5": None if pd.isna(r.ws100_era5) else round(float(r.ws100_era5), 2), "status": r.status}
                         for t, r in df.iterrows()]
    return out


# --------------------------------------------------------------------------- backtest
@lru_cache(maxsize=1)
def _backtest_table() -> pd.DataFrame:
    p = get_settings().processed_dir / "kelmarsh_backtest.parquet"
    if not p.exists():
        raise ServiceError("Бэктест нәтижелері жоқ. scripts/04_run_experiments.py іске қосыңыз.")
    return pd.read_parquet(p)


def _bt_period(farm, start_local, end_local):
    bt = _backtest_table()
    lo, hi = bt.valid_time.min(), bt.valid_time.max() + pd.Timedelta(hours=1)
    s = parse_local(start_local, farm, lo)
    e = parse_local(end_local, farm, hi)
    sub = bt[(bt.valid_time >= s) & (bt.valid_time < e)]
    if sub.empty:
        raise ServiceError(f"Бэктест тек тест кезеңінде қолжетімді: {fmt_local(lo, farm)} – {fmt_local(hi, farm)}.")
    return sub, f"{fmt_local(sub.valid_time.min(), farm)} – {fmt_local(sub.valid_time.max() + pd.Timedelta(hours=1), farm)}"


def backtest(farm_id: str, model: str = "ensemble", start_local: str | None = None, end_local: str | None = None,
             horizon: int = 24) -> dict:
    farm = _require_scada(farm_id)
    sub, period = _bt_period(farm, start_local, end_local)
    hs = sorted(sub.h.unique())
    if horizon not in hs:
        raise ServiceError(f"Горизонт {horizon} сағ бэктестте жоқ. Қолжетімді: {hs}.")
    sub = sub[sub.h == horizon]
    if model not in set(sub.model):
        raise ServiceError(f"'{model}' моделі бойынша бэктест жоқ. Қолжетімді: {sorted(set(sub.model))}.")
    t = metrics_table(sub[sub.model.isin([model, "persistence"])])
    row = t[t.model == model].iloc[0]
    return {
        "farm": farm.name, "model": model, "model_name": MODEL_NAMES.get(model, model), "horizon_h": horizon,
        "period_local": period, "unit": "MW",
        "metrics": {
            "mae_mw": round(float(row.mae), 3), "rmse_mw": round(float(row.rmse), 3),
            "nmae_pct": round(float(row.nmae), 2), "nrmse_pct": round(float(row.nrmse), 2),
            "bias_mw": round(float(row.bias), 3),
            "skill_vs_persistence": None if pd.isna(row.skill) else round(float(row.skill), 3),
            "crps_pct": round(float(row.crps) * 100, 2), "picp_80_pct": round(float(row.picp_80), 1),
            "n_hours": int(row.n),
        },
    }


def compare_models(farm_id: str, start_local: str | None = None, end_local: str | None = None,
                   horizon: int = 24) -> dict:
    farm = _require_scada(farm_id)
    sub, period = _bt_period(farm, start_local, end_local)
    sub = sub[sub.h == horizon]
    if sub.empty:
        raise ServiceError(f"Горизонт {horizon} сағ бэктестте жоқ. Қолжетімді: {sorted(_backtest_table().h.unique())}.")
    research = ("lstm", "tft", "chronos")
    t = metrics_table(sub[~sub.model.isin(research)])
    if sub.model.isin(research).any():  # зерттеу модельдері — өз жиынтығында (күнделікті болжам)
        tr = metrics_table(sub)
        t = pd.concat([t, tr[tr.model.isin(research)]], ignore_index=True)
    t = t.sort_values("nmae")
    return {
        "farm": farm.name, "horizon_h": horizon, "period_local": period,
        "best_model": t[~t.model.isin(research)].iloc[0].model,
        "note": "LSTM/TFT/Chronos тек күнделікті 00:00 UTC болжамдарында бағаланған (n_hours бағанын қараңыз)",
        "table": [{
            "model": r.model, "model_name": MODEL_NAMES.get(r.model, r.model),
            "nmae_pct": round(float(r.nmae), 2), "nrmse_pct": round(float(r.nrmse), 2),
            "mae_mw": round(float(r.mae), 3), "skill_vs_persistence": None if pd.isna(r.skill) else round(float(r.skill), 3),
            "crps_pct": round(float(r.crps) * 100, 2), "picp_80_pct": round(float(r.picp_80), 1),
            "n_hours": int(r.n),
        } for r in t.itertuples()],
        "n_hours": int(t[~t.model.isin(research)].n.max()),
    }


def backtest_series(farm_id: str, model: str, horizon: int, start_local=None, end_local=None, max_points=24 * 14) -> dict:
    farm = _require_scada(farm_id)
    sub, period = _bt_period(farm, start_local, end_local)
    sub = sub[(sub.h == horizon) & (sub.model == model)].sort_values("valid_time").tail(max_points)
    cap = farm.capacity_mw
    return {"farm": farm.name, "model": model, "horizon_h": horizon, "series": [
        {"time_local": fmt_local(r.valid_time, farm), "actual": None if pd.isna(r.y) else round(r.y * cap, 3),
         "p10": round(float(r["q0.1"]) * cap, 3),
         "p50": round(float(r["q0.5"]) * cap, 3), "p90": round(float(r["q0.9"]) * cap, 3)}
        for _, r in sub.iterrows()]}


# --------------------------------------------------------------------------- downtime
def detect_downtime(farm_id: str, start_local: str | None = None, end_local: str | None = None,
                    min_duration_h: int = 2) -> dict:
    """Жел жеткілікті (физикалық модель бойынша күтілетін өндіріс ≥ 25% P_inst), ал нақты
    өндіріс < 5% P_inst болған сағаттар -> тоқтап тұру/шектеу кезеңдері."""
    from app.features.physics import density_corrected_speed, hub_speed, shear_exponent
    from app.models.registry import load_bundle

    farm = _require_scada(farm_id)
    s, e, rng = _period(farm, start_local, end_local, default_days=30)
    df = _load_history(farm_id, s, e)
    if df.empty:
        raise ServiceError(f"Бұл кезеңге деректер жоқ. Қолжетімді кезең: {rng['start_local']} – {rng['end_local']}.")
    with session() as ss:
        wh = pd.DataFrame(ss.execute(select(WeatherHistory.timestamp, WeatherHistory.wind_speed_10m,
                                            WeatherHistory.wind_speed_100m, WeatherHistory.temperature, WeatherHistory.pressure)
                                     .where(WeatherHistory.farm_id == farm_id, WeatherHistory.timestamp >= s,
                                            WeatherHistory.timestamp < e)).all(),
                          columns=["time", "ws10", "ws100", "t2m", "sp"]).set_index("time")
    curve = load_bundle(farm_id).curve
    alpha = shear_exponent(wh.ws10, 10, wh.ws100, 100)
    rho = wh.sp * 100 / (287.05 * (wh.t2m + 273.15))
    expected = pd.Series(curve(density_corrected_speed(hub_speed(wh.ws100, 100, farm.hub_height_m, alpha), rho)),
                         index=wh.index) * farm.capacity_mw
    df["expected_mw"] = expected.reindex(df.index)
    actual = df.raw_power_mw
    flag = (df.expected_mw >= 0.25 * farm.capacity_mw) & ((actual < 0.05 * farm.capacity_mw) | actual.isna() & (df.status == "outage"))
    grp = (flag.astype(int).diff().fillna(0) != 0).cumsum()
    periods = []
    for _, g in df[flag].groupby(grp[flag]):
        if len(g) >= min_duration_h:
            periods.append({
                "start_local": fmt_local(g.index[0], farm),
                "end_local": fmt_local(g.index[-1] + pd.Timedelta(hours=1), farm),
                "duration_h": int(len(g)),
                "mean_ws100_era5": round(float(g.ws100_era5.mean()), 2),
                "expected_energy_mwh": round(float(g.expected_mw.sum()), 1),
                "actual_energy_mwh": round(float(g.raw_power_mw.fillna(0).clip(lower=0).sum()), 1),
                "lost_energy_mwh": round(float((g.expected_mw - g.raw_power_mw.fillna(0).clip(lower=0)).clip(lower=0).sum()), 1),
            })
    return {
        "farm": farm.name, "period_local": f"{fmt_local(s, farm)} – {fmt_local(e, farm)}",
        "criterion": "күтілетін қуат ≥ 25% P_inst (ERA5 + қуат қисығы), нақты қуат < 5% P_inst",
        "min_duration_h": min_duration_h,
        "n_periods": len(periods),
        "total_hours": int(sum(p["duration_h"] for p in periods)),
        "total_lost_energy_mwh": round(float(sum(p["lost_energy_mwh"] for p in periods)), 1),
        "periods": periods[:50],
    }
