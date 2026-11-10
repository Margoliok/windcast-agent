"""SCADA деректерін тазалау және станцияның сағаттық қуатын құрастыру.

Ережелер (2.1-бөлім):
  1. Бос мәндер (NaN) алынады.
  2. Тоқтап тұру: Status файлындағы 'Stop' оқиғалары жабатын интервалдар.
  3. Тоқтап тұру (жанама): жел cut-in-нен жоғары (>4.5 м/с), ал қуат ≈ 0.
  4. Қуатты шектеу (curtailment): қуат орнатпасы (setpoint) номиналдан төмен,
     немесе номиналдан төмен желде қалақ бұрышы (pitch) үлкен.
  5. Шығарындылар: эмпирикалық қисықтан ауытқу > max(15% P_rated, 4·MAD бин ішінде).
Таза 10-мин нүктелер сағат белгісіне орталанған терезеге [t−30, t+30) агрегацияланады; турбина-сағат таза болып саналады, егер
6 интервалдың кемінде 4-еуі таза болса. Станция сағаты: кемінде 5/6 турбина таза болса,
қуат толық қолжетімділікке масштабталады (P_station = ΣP_clean · N/N_clean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.physics import PowerCurve
from app.ingest.kelmarsh import hourly_centered, stop_mask

RATED_KW = 2050.0
CUT_IN = 4.5


def clean_turbine(data: pd.DataFrame, status: pd.DataFrame | None, rated_kw: float = RATED_KW) -> pd.DataFrame:
    """10 минуттық турбина деректеріне сапа белгілерін қосады."""
    df = data.copy()
    df["f_missing"] = df["power_kw"].isna() | df["ws"].isna()
    df["f_stop"] = stop_mask(df.index, status) & ~df["f_missing"]
    df["f_zero_at_wind"] = (df["ws"] > CUT_IN) & (df["power_kw"] <= 0.01 * rated_kw) & ~df["f_missing"] & ~df["f_stop"]

    curt = pd.Series(False, index=df.index)
    if "setpoint_kw" in df:
        curt |= df["setpoint_kw"] < 0.95 * rated_kw
    if "pitch" in df:
        curt |= (df["ws"] < 11.0) & (df["pitch"] > 6.0) & (df["power_kw"] < 0.9 * rated_kw)
    df["f_curtail"] = curt & ~df["f_missing"] & ~df["f_stop"] & ~df["f_zero_at_wind"]

    ok = ~(df.f_missing | df.f_stop | df.f_zero_at_wind | df.f_curtail)
    df["f_outlier"] = False
    if ok.sum() > 500:
        pn = df.loc[ok, "power_kw"] / rated_kw
        curve = PowerCurve.fit(df.loc[ok, "ws"], pn)
        resid = pn - curve(df.loc[ok, "ws"])
        bins = (df.loc[ok, "ws"] // 0.5).astype(int)
        mad = resid.groupby(bins).transform(lambda r: np.median(np.abs(r - np.median(r))))
        thr = np.maximum(0.15, 4 * 1.4826 * mad)
        df.loc[ok, "f_outlier"] = np.abs(resid) > thr
    df["clean"] = ok & ~df["f_outlier"]
    return df


def quality_row(turbine: int, df: pd.DataFrame) -> dict:
    """Кесте 7: турбина бойынша сапа көрсеткіштері (%)."""
    n = len(df)
    pct = lambda s: round(100.0 * s.sum() / n, 2) if n else 0.0  # noqa: E731
    return {
        "turbine": turbine,
        "records": n,
        "missing_pct": pct(df.f_missing),
        "stop_pct": pct(df.f_stop),
        "zero_at_wind_pct": pct(df.f_zero_at_wind),
        "curtail_pct": pct(df.f_curtail),
        "outlier_pct": pct(df.f_outlier),
        "clean_pct": pct(df.clean),
    }


def station_hourly(turbines: dict[int, pd.DataFrame], rated_kw: float = RATED_KW, min_frac: float = 5 / 6) -> pd.DataFrame:
    """Таза турбина деректерінен станцияның сағаттық қуаты (МВт)."""
    n_turb = len(turbines)
    p_clean, ok_cols, ws_cols, raw_cols = {}, {}, {}, {}
    for t, df in turbines.items():
        p_clean[t] = hourly_centered(df["power_kw"].where(df["clean"]), "mean")
        ok_cols[t] = hourly_centered(df["clean"].astype(int), "sum") >= 4
        ws_cols[t] = hourly_centered(df["ws"].where(df["clean"]), "mean")
        raw_cols[t] = hourly_centered(df["power_kw"], "mean")
    P = pd.DataFrame(p_clean)
    OK = pd.DataFrame(ok_cols).fillna(False).astype(bool)
    n_ok = OK.sum(axis=1)
    p_sum = P.where(OK).sum(axis=1, min_count=1)
    out = pd.DataFrame(index=P.index)
    out["availability"] = n_ok / n_turb
    out["power_mw"] = (p_sum * n_turb / n_ok.replace(0, np.nan) / 1000.0).where(out["availability"] >= min_frac)
    out["power_raw_mw"] = pd.DataFrame(raw_cols).sum(axis=1, min_count=n_turb) / 1000.0
    out["nacelle_ws"] = pd.DataFrame(ws_cols).mean(axis=1)
    out["power_mw"] = out["power_mw"].clip(0, n_turb * rated_kw / 1000.0)
    return out
