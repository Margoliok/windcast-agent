"""Зерттеу деректер жиынтығын құрастыру: станция қуаты + үш түрлі ауа райы дереккөзі.

Екі режим (қолжетімді деректерге қарай автоматты таңдалады):
  * "scada"      — Kelmarsh SCADA 2020–2024 (толық, ~2.7 ГБ, этап B). Турбина деңгейінде тазалау.
                   train 2020–2022, val 2023, test 2024. Test жылы үшін Previous Runs API
                   арқылы нақты 1/2/3 тәулік бұрынғы болжамдар алынады.
  * "grid_meter" — станция есептегіші 2016–2019 (шағын файл, этап A). Тек станция деңгейіндегі
                   тазалау. train 2017-01…2018-06, val 2018 II жартыжылдық, test 2019-01…2019-09
                   (NWP архиві 2017 жылдан басталады).
                   Бұл режимде горизонтқа тәуелді NWP архиві жоқ — нәтижелер алдын ала (черновик).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.farms import KELMARSH
from app.ingest import openmeteo
from app.ingest.cleaning import clean_turbine, quality_row, station_hourly
from app.ingest.kelmarsh import read_grid_meter, read_scada_zip


@dataclass
class Splits:
    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]


SPLITS = {
    "scada": Splits(("2020-01-01", "2022-12-31 23:00"), ("2023-01-01", "2023-12-31 23:00"), ("2024-01-01", "2024-12-31 23:00")),
    "grid_meter": Splits(("2017-01-01", "2018-06-30 23:00"), ("2018-07-01", "2018-12-31 23:00"), ("2019-01-01", "2019-09-30 23:00")),
}


def kelmarsh_dir() -> Path:
    return get_settings().raw_dir / "kelmarsh"


def scada_zips() -> list[Path]:
    zips = sorted(kelmarsh_dir().glob("Kelmarsh_SCADA_20*.zip"))
    return [z for z in zips if 2020 <= int(z.name.split("_")[2]) <= 2024]


def detect_mode() -> str:
    return "scada" if len(scada_zips()) >= 3 else "grid_meter"


def processed(name: str) -> Path:
    return get_settings().processed_dir / name


# --------------------------------------------------------------------------- power
def build_power_scada(log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> (сағаттық станция қуаты, Кесте 7 сапа көрсеткіштері, Сурет 5 үшін 10-мин үлгі)."""
    hourly, quality, samples = [], [], []
    per_turbine: dict[int, list[pd.DataFrame]] = {}
    for z in scada_zips():
        log(f"  SCADA: {z.name}")
        for t, parts in read_scada_zip(z).items():
            if "data" not in parts:
                continue
            per_turbine.setdefault(t, []).append(clean_turbine(parts["data"], parts.get("status")))
    turbines = {t: pd.concat(v).sort_index() for t, v in per_turbine.items()}
    turbines = {t: df[~df.index.duplicated()] for t, df in turbines.items()}
    for t, df in sorted(turbines.items()):
        quality.append(quality_row(t, df))
        samples.append(df[["ws", "power_kw", "clean"]].sample(min(len(df), 30000), random_state=t).assign(turbine=t))
    hourly = station_hourly(turbines)
    q = pd.DataFrame(quality)
    tot = {"turbine": "Барлығы", "records": int(q.records.sum())}
    for c in q.columns[2:]:
        tot[c] = round(float(np.average(q[c], weights=q.records)), 2)
    q = pd.concat([q, pd.DataFrame([tot])], ignore_index=True)
    return hourly, q, pd.concat(samples)


def build_power_grid_meter() -> tuple[pd.DataFrame, pd.DataFrame]:
    gm = read_grid_meter(kelmarsh_dir() / "Kelmarsh_Grid_Meter_4458.zip")
    cap = KELMARSH.capacity_mw
    out = pd.DataFrame(index=gm.index)
    out["power_raw_mw"] = gm["power_mw"]
    bad = (gm["power_mw"] < -0.05 * cap) | (gm["power_mw"] > 1.02 * cap)
    out["power_mw"] = gm["power_mw"].where(~bad).clip(0, cap)
    out["availability"] = np.nan  # есептегіш турбиналардың күйін бермейді
    n = len(out.loc["2016-05-01":"2019-12-31"])
    sub = out.loc["2016-05-01":"2019-12-31"]
    quality = pd.DataFrame([{
        "turbine": "Grid meter",
        "records": n,
        "missing_pct": round(100 * sub.power_raw_mw.isna().mean(), 2),
        "invalid_pct": round(100 * bad.loc[sub.index].mean(), 2),
    }])
    return out, quality


def flag_outages(power: pd.DataFrame, wx: pd.DataFrame, cap: float) -> pd.Series:
    """Есептегіш режимі: жел күшті (ERA5 ws100 > 8 м/с), ал өндіріс < 2% — тоқтап тұру/шектеу."""
    ws = wx["ws100"].reindex(power.index)
    return (ws > 8.0) & (power["power_mw"] < 0.02 * cap)


# --------------------------------------------------------------------------- weather
def fetch_weather(start: str, end: str, test_start: str, test_end: str, mode: str, log=print) -> dict[str, pd.DataFrame]:
    f = KELMARSH
    log("  Open-Meteo: тарихи NWP болжамдары")
    wx = {"nwp": openmeteo.fetch_historical_forecast(f.lat, f.lon, start, end)}
    log("  Open-Meteo: ERA5 реанализі")
    wx["era5"] = openmeteo.fetch_era5(f.lat, f.lon, start, end)
    if mode == "scada":
        for d in (1, 2, 3):
            log(f"  Open-Meteo: Previous Runs, {d} тәулік бұрын")
            wx[f"prev{d}"] = openmeteo.fetch_previous_runs(f.lat, f.lon, test_start, test_end, d)
    return wx


# --------------------------------------------------------------------------- main
def build(mode: str | None = None, log=print) -> dict:
    mode = mode or detect_mode()
    sp = SPLITS[mode]
    log(f"Режим: {mode}")
    if mode == "scada":
        power, quality, sample = build_power_scada(log)
        sample.to_parquet(processed("kelmarsh_scada_sample.parquet"))
    else:
        power, quality = build_power_grid_meter()
    power = power.loc[sp.train[0]: sp.test[1]]
    wx = fetch_weather(sp.train[0][:10], sp.test[1][:10], sp.test[0][:10], sp.test[1][:10], mode, log)
    if mode == "grid_meter":
        out = flag_outages(power, wx["era5"], KELMARSH.capacity_mw)
        power["f_outage"] = out
        power.loc[out, "power_mw"] = np.nan
        quality["outage_pct"] = round(100 * out.mean(), 2)
        quality["clean_pct"] = round(100 * power["power_mw"].notna().mean(), 2)
    power["power_norm"] = power["power_mw"] / KELMARSH.capacity_mw
    power.to_parquet(processed("kelmarsh_power.parquet"))
    quality.to_csv(processed("kelmarsh_quality.csv"), index=False)
    for k, v in wx.items():
        v.to_parquet(processed(f"kelmarsh_wx_{k}.parquet"))
    meta = {
        "mode": mode,
        "splits": asdict(sp),
        "weather_sources": list(wx),
        "n_hours": int(len(power)),
        "n_valid_hours": int(power["power_norm"].notna().sum()),
        "capacity_mw": KELMARSH.capacity_mw,
    }
    processed("dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(meta, ensure_ascii=False))
    return meta


def load_meta() -> dict:
    return json.loads(processed("dataset_meta.json").read_text(encoding="utf-8"))


def load_power() -> pd.DataFrame:
    return pd.read_parquet(processed("kelmarsh_power.parquet"))


def load_weather(source: str) -> pd.DataFrame:
    return pd.read_parquet(processed(f"kelmarsh_wx_{source}.parquet"))


def available_weather() -> list[str]:
    return [p.stem.replace("kelmarsh_wx_", "") for p in get_settings().processed_dir.glob("kelmarsh_wx_*.parquet")]
