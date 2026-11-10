"""Болжам сервистері — API, ЖИ-агент құралдары және жоспарлаушы үшін ортақ логика.

Барлық сандар осында есептеледі (solver-grounded): LLM тек осы функциялардың нәтижесін
түсіндіреді. Ішкі уақыт — UTC; пайдаланушыға станцияның жергілікті уақыты көрсетіледі.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.agent.gate import check_series
from app.core.farms import Farm, get_farm
from app.features.build import weather_features
from app.ingest import openmeteo
from app.models.registry import LIVE_MODELS, load_bundle

_WX_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
WX_TTL_S = 15 * 60


class ServiceError(ValueError):
    """Пайдаланушыға түсінікті қате (агент оны ашық хабарлайды)."""


# --------------------------------------------------------------------------- time helpers
def to_local(ts: pd.DatetimeIndex | pd.Timestamp, farm: Farm):
    if isinstance(ts, pd.Timestamp):
        return (ts.tz_localize("UTC") if ts.tzinfo is None else ts).tz_convert(farm.timezone)
    return ts.tz_localize("UTC").tz_convert(farm.timezone)


def parse_local(value: str | None, farm: Farm, default: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """'2026-11-11 10:00' (жергілікті уақыт) -> UTC naive Timestamp."""
    if value in (None, ""):
        return default
    try:
        ts = pd.Timestamp(value)
    except Exception as e:
        raise ServiceError(f"Уақыт форматы қате: {value!r}. Мысалы: 2026-11-11 10:00") from e
    if ts.tzinfo is None:
        ts = ts.tz_localize(farm.timezone)
    return ts.tz_convert("UTC").tz_localize(None)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")


def fmt_local(ts_utc: pd.Timestamp, farm: Farm) -> str:
    return to_local(ts_utc, farm).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- weather
def live_weather(farm: Farm, hours: int = 72) -> pd.DataFrame:
    key = farm.id
    hit = _WX_CACHE.get(key)
    if hit and time.time() - hit[0] < WX_TTL_S and hit[1].attrs.get("hours", 0) >= hours:
        return hit[1]
    try:
        wx = openmeteo.fetch_live_forecast(farm.lat, farm.lon, hours=max(hours, 72), past_hours=48)
    except Exception as e:
        raise ServiceError(f"Open-Meteo ауа райы болжамын алу мүмкін болмады: {e}") from e
    wx.attrs["hours"] = max(hours, 72)
    _WX_CACHE[key] = (time.time(), wx)
    return wx


def _features(farm: Farm, wx: pd.DataFrame, wind_scale: float = 1.0) -> pd.DataFrame:
    bundle = load_bundle(farm.id)
    wx = wx.copy()
    if wind_scale != 1.0:
        for c in ("ws10", "ws80", "ws100", "ws120", "gust10"):
            if c in wx:
                wx[c] = wx[c] * wind_scale
    return weather_features(wx, farm.hub_height_m, bundle.curve)


# --------------------------------------------------------------------------- forecast
@dataclass
class Forecast:
    farm: Farm
    model: str
    issue_time: pd.Timestamp
    frame: pd.DataFrame  # index: valid_time UTC; p10/p50/p90 МВт; ws_hub, ws100, wd100, t2m
    wind_scale: float = 1.0

    def rows(self) -> list[dict]:
        out = []
        loc = to_local(self.frame.index, self.farm)
        for (ts, r), tl in zip(self.frame.iterrows(), loc):
            out.append({
                "time_local": tl.strftime("%Y-%m-%d %H:%M"),
                "time_utc": ts.strftime("%Y-%m-%dT%H:%MZ"),
                "p10": round(float(r.p10), 3), "p50": round(float(r.p50), 3), "p90": round(float(r.p90), 3),
                "ws_hub": round(float(r.ws_hub), 2), "ws100": round(float(r.ws100), 2),
                "wd100": None if pd.isna(r.wd100) else int(round(float(r.wd100))),
            })
        return out


def run_forecast(farm_id: str, hours: int = 72, model: str | None = None, wind_scale: float = 1.0) -> Forecast:
    farm = get_farm(farm_id)
    if not 1 <= hours <= 168:
        raise ServiceError("Горизонт 1–168 сағат аралығында болуы керек (негізгі режим — 72 сағат).")
    bundle = load_bundle(farm.id)
    model = model or bundle.best_model
    if model not in bundle.available_models:
        raise ServiceError(f"'{model}' моделі жұмыс режимінде қолжетімсіз. Қолжетімді: {', '.join(bundle.available_models)}")
    wx = live_weather(farm, hours)
    feats = _features(farm, wx, wind_scale)
    issue = now_utc()
    valid = feats.loc[(feats.index > issue) & (feats.index <= issue + pd.Timedelta(hours=hours))]
    valid = valid.dropna(subset=["ws100"])
    q = bundle.predict_nwp(valid, model) * farm.capacity_mw
    frame = pd.DataFrame({"p10": q["q0.1"], "p50": q["q0.5"], "p90": q["q0.9"]}, index=valid.index)
    frame["ws_hub"] = valid["ws_hub"]
    frame["ws100"] = valid["ws100"]
    frame["wd100"] = wx["wd100"].reindex(valid.index) if "wd100" in wx else np.nan
    frame["t2m"] = valid["t2m"]
    return Forecast(farm, model, issue, frame, wind_scale)


def summarize(fc: Forecast, start_utc: pd.Timestamp | None = None, end_utc: pd.Timestamp | None = None) -> dict:
    """Кезеңдегі энергия (МВт·сағ) және статистика. Шекаралар: [start, end) жергілікті сағаттар."""
    f = fc.frame
    if start_utc is not None:
        f = f[f.index >= start_utc]
    if end_utc is not None:
        f = f[f.index < end_utc]
    if f.empty:
        raise ServiceError("Көрсетілген кезең болжам горизонтына кірмейді.")
    cap = fc.farm.capacity_mw
    imax, imin = f.p50.idxmax(), f.p50.idxmin()
    return {
        "period_local": f"{fmt_local(f.index[0], fc.farm)} – {fmt_local(f.index[-1] + pd.Timedelta(hours=1), fc.farm)}",
        "hours": int(len(f)),
        "energy_mwh_p50": round(float(f.p50.sum()), 1),
        "energy_mwh_p10": round(float(f.p10.sum()), 1),
        "energy_mwh_p90": round(float(f.p90.sum()), 1),
        "mean_mw_p50": round(float(f.p50.mean()), 2),
        "max_mw_p50": round(float(f.p50.max()), 2),
        "max_time_local": fmt_local(imax, fc.farm),
        "min_mw_p50": round(float(f.p50.min()), 2),
        "min_time_local": fmt_local(imin, fc.farm),
        "capacity_factor_pct": round(float(f.p50.mean() / cap * 100), 1),
        "mean_ws_hub": round(float(f.ws_hub.mean()), 2),
    }


def forecast_payload(fc: Forecast, start_utc=None, end_utc=None, include_series: bool = True) -> dict:
    rows = fc.rows()
    gate = check_series(rows, fc.farm.capacity_mw, expected_len=len(fc.frame), unit="MW")
    bundle = load_bundle(fc.farm.id)
    out = {
        "farm": fc.farm.name,
        "farm_id": fc.farm.id,
        "model": fc.model,
        "issue_time_local": fmt_local(fc.issue_time, fc.farm),
        "timezone": fc.farm.timezone,
        "unit": "MW",
        "capacity_mw": fc.farm.capacity_mw,
        "summary": summarize(fc, start_utc, end_utc),
        "gate": gate.to_dict(),
        "model_info": {"best_model": bundle.best_model, "data_mode": bundle.meta.get("mode")},
    }
    if fc.wind_scale != 1.0:
        out["wind_scale"] = fc.wind_scale
    if include_series:
        out["series"] = rows
    return out


def predict_power(farm_id: str, hours: int = 72, model: str | None = None, start_local: str | None = None,
                  end_local: str | None = None) -> dict:
    farm = get_farm(farm_id)
    fc = run_forecast(farm_id, hours, model)
    return forecast_payload(fc, parse_local(start_local, farm), parse_local(end_local, farm))


# --------------------------------------------------------------------------- weather tool
def weather_forecast(farm_id: str, hours: int = 72, low_wind_threshold: float | None = None,
                     min_duration_h: int = 6) -> dict:
    farm = get_farm(farm_id)
    wx = live_weather(farm, hours)
    feats = _features(farm, wx)
    issue = now_utc()
    f = feats.loc[(feats.index > issue) & (feats.index <= issue + pd.Timedelta(hours=hours))]
    loc = to_local(f.index, farm)
    series = [{
        "time_local": tl.strftime("%Y-%m-%d %H:%M"),
        "ws10": round(float(r.ws10), 2), "ws100": round(float(r.ws100), 2), "ws_hub": round(float(r.ws_hub), 2),
        "gust10": None if pd.isna(r.gust10) else round(float(r.gust10), 2),
        "wd100": None if pd.isna(wx.wd100.get(ts)) else int(round(float(wx.wd100.get(ts)))),
        "t2m": round(float(r.t2m), 1), "sp_hpa": round(float(r.sp), 1), "rho": round(float(r.rho), 3),
    } for (ts, r), tl in zip(f.iterrows(), loc)]
    out = {
        "farm": farm.name, "timezone": farm.timezone, "hours": len(series), "source": "Open-Meteo Forecast API",
        "hub_height_m": farm.hub_height_m,
        "summary": {
            "mean_ws_hub": round(float(f.ws_hub.mean()), 2), "max_ws_hub": round(float(f.ws_hub.max()), 2),
            "min_ws_hub": round(float(f.ws_hub.min()), 2), "max_gust10": round(float(f.gust10.max()), 2),
            "mean_t2m": round(float(f.t2m.mean()), 1),
        },
        "series": series,
    }
    if low_wind_threshold is not None:
        out["low_wind_windows"] = low_wind_windows(f, farm, low_wind_threshold, min_duration_h)
        out["low_wind_threshold_ms"] = low_wind_threshold
        out["min_duration_h"] = min_duration_h
    return out


def low_wind_windows(f: pd.DataFrame, farm: Farm, threshold: float, min_duration_h: int) -> list[dict]:
    """Ступица биіктігіндегі жел < threshold болатын, ұзақтығы ≥ min_duration_h үздіксіз кезеңдер."""
    low = (f["ws_hub"] < threshold).astype(int)
    grp = (low.diff().fillna(0) != 0).cumsum()
    out = []
    for _, g in f[low == 1].groupby(grp[low == 1]):
        if len(g) >= min_duration_h:
            out.append({
                "start_local": fmt_local(g.index[0], farm),
                "end_local": fmt_local(g.index[-1] + pd.Timedelta(hours=1), farm),
                "duration_h": int(len(g)),
                "mean_ws_hub": round(float(g.ws_hub.mean()), 2),
            })
    return out


# --------------------------------------------------------------------------- what-if
def what_if(farm_id: str, delta_wind_pct: float, hours: int = 24, model: str | None = None,
            start_local: str | None = None, end_local: str | None = None) -> dict:
    if not -60 <= delta_wind_pct <= 60:
        raise ServiceError("Жел жылдамдығының өзгерісі −60% … +60% аралығында болуы керек.")
    farm = get_farm(farm_id)
    s, e = parse_local(start_local, farm), parse_local(end_local, farm)
    base = run_forecast(farm_id, hours, model)
    scen = run_forecast(farm_id, hours, model, wind_scale=1 + delta_wind_pct / 100.0)
    sb, ss = summarize(base, s, e), summarize(scen, s, e)
    gate = check_series(scen.rows(), farm.capacity_mw, len(scen.frame))
    return {
        "farm": farm.name, "model": base.model, "delta_wind_pct": delta_wind_pct,
        "baseline": sb, "scenario": ss,
        "delta_energy_mwh_p50": round(ss["energy_mwh_p50"] - sb["energy_mwh_p50"], 1),
        "delta_energy_pct": round((ss["energy_mwh_p50"] / sb["energy_mwh_p50"] - 1) * 100, 1) if sb["energy_mwh_p50"] else None,
        "note": "Жел жылдамдығы барлық биіктікте бірдей масштабталады; қуат кубтық тәуелділікпен және қуат қисығымен шектеледі.",
        "gate": gate.to_dict(),
        "series": [{"time_local": a["time_local"], "p50_base": a["p50"], "p50_scenario": b["p50"]}
                   for a, b in zip(base.rows(), scen.rows())],
    }


# --------------------------------------------------------------------------- explanation
def explain_forecast(farm_id: str, start_local: str | None = None, end_local: str | None = None,
                     reference_start_local: str | None = None, reference_end_local: str | None = None,
                     lang: str = "kk") -> dict:
    from app.explain.shap_explain import explain_window

    farm = get_farm(farm_id)
    bundle = load_bundle(farm_id)
    wx = live_weather(farm, 72)
    feats = _features(farm, wx)
    issue = now_utc()
    fut = feats.loc[(feats.index > issue) & (feats.index <= issue + pd.Timedelta(hours=72))].dropna(subset=["ws100"])
    s = parse_local(start_local, farm, fut.index[0])
    e = parse_local(end_local, farm, fut.index[-1] + pd.Timedelta(hours=1))
    win = fut[(fut.index >= s) & (fut.index < e)]
    if win.empty:
        raise ServiceError("Түсіндіру кезеңі болжам горизонтына (72 сағ) кірмейді.")
    ref = None
    rs, re_ = parse_local(reference_start_local, farm), parse_local(reference_end_local, farm)
    if rs is not None and re_ is not None:
        ref = fut[(fut.index >= rs) & (fut.index < re_)]
    elif start_local:  # әдепкі салыстыру: дәл алдындағы тең ұзындықты кезең
        dur = e - s
        ref = feats[(feats.index >= s - dur) & (feats.index < s)].dropna(subset=["ws100"])
    res = explain_window(bundle, win, ref if ref is not None and len(ref) else None, lang=lang)
    res.update({
        "farm": farm.name,
        "window_local": f"{fmt_local(win.index[0], farm)} – {fmt_local(win.index[-1] + pd.Timedelta(hours=1), farm)}",
        "reference_local": None if ref is None or not len(ref) else
        f"{fmt_local(ref.index[0], farm)} – {fmt_local(ref.index[-1] + pd.Timedelta(hours=1), farm)}",
        "window_mean_ws_hub": round(float(win.ws_hub.mean()), 2),
        "reference_mean_ws_hub": None if ref is None or not len(ref) else round(float(ref.ws_hub.mean()), 2),
    })
    return res


def model_list(farm_id: str) -> dict:
    b = load_bundle(farm_id)
    return {"farm_id": farm_id, "live_models": [m for m in b.available_models if m in LIVE_MODELS or m in b.nwp_models],
            "best_model": b.best_model, "ensemble_weights": b.ensemble_weights, "meta": b.meta}
