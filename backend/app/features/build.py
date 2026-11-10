"""Модель белгілерін қалыптастыру (Кесте 8).

Екі режим:
  * NWP режимі (горизонт > 6 сағ, тәулік алдындағы болжам) — тек жарамды уақыттағы (valid time)
    ауа райы болжамы мен уақыт белгілері. Модель горизонтқа тәуелсіз; горизонттың әсері кіріс
    болжамның сапасы арқылы көрінеді (алдын ала болжау уақыты көп болса, NWP қателігі көп).
  * Өте қысқа мерзімді режим (1–6 сағ) — NWP белгілеріне қоса, болжам шығарылған сәттегі
    (issue time) нақты қуат пен оның лагтары, горизонт h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.physics import (
    PowerCurve,
    air_density,
    density_corrected_speed,
    hub_speed,
    shear_exponent,
    wind_components,
)

SHORT_HORIZONS = tuple(range(1, 7))

NWP_FEATURES = [
    "ws10", "ws80", "ws100", "ws120", "gust10", "gust_factor",
    "u100", "v100", "wd_sin", "wd_cos",
    "t2m", "sp", "rho", "alpha", "ws_hub", "ws_hub_rho", "p_phys",
    "ws_hub_d1", "ws_hub_std3", "ws_hub_prev", "ws_hub_next",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
SHORT_FEATURES = NWP_FEATURES + ["h", "p_now", "p_lag1", "p_lag2", "p_mean3", "p_phys_now", "resid_now"]

# Кесте 8 үшін белгілердің сипаттамасы: (топ, белгі, дереккөз, өлшем бірлігі)
FEATURE_TABLE = [
    ("Метеорологиялық", "ws10, ws80, ws100, ws120 — 10/80/100/120 м биіктіктегі жел жылдамдығы", "Open-Meteo NWP", "м/с"),
    ("Метеорологиялық", "gust10 — 10 м биіктіктегі екпін", "Open-Meteo NWP", "м/с"),
    ("Метеорологиялық", "t2m — ауа температурасы, sp — жер бетіндегі қысым", "Open-Meteo NWP", "°C, гПа"),
    ("Туынды физикалық", "rho — ауа тығыздығы, (6) формула", "есептелген", "кг/м³"),
    ("Туынды физикалық", "alpha — желдің ығысу көрсеткіші (10–100 м)", "есептелген", "—"),
    ("Туынды физикалық", "ws_hub — ступица биіктігіндегі жылдамдық, (7) формула", "есептелген", "м/с"),
    ("Туынды физикалық", "ws_hub_rho — тығыздыққа түзетілген жылдамдық", "есептелген", "м/с"),
    ("Туынды физикалық", "u100, v100, wd_sin, wd_cos — бағыт компоненттері, (8) формула", "есептелген", "м/с, —"),
    ("Туынды физикалық", "p_phys — қуат қисығы бойынша «физикалық болжам», (10) формула", "есептелген", "о.б. (0–1)"),
    ("Туынды физикалық", "gust_factor, ws_hub_d1, ws_hub_std3, ws_hub_prev/next — екпін коэффициенті және болжам траекториясы", "есептелген", "—, м/с"),
    ("Уақыттық", "hour_sin/cos, doy_sin/cos — тәулік сағаты, жыл күні", "күнтізбе", "—"),
    ("Тарихи (1–6 сағ)", "p_now, p_lag1, p_lag2, p_mean3 — болжам сәтіндегі нақты қуат және лагтар", "SCADA", "о.б."),
    ("Тарихи (1–6 сағ)", "p_phys_now, resid_now, h — физикалық модельдің ағымдағы қателігі, горизонт", "SCADA + NWP", "о.б., сағ"),
]


def weather_features(wx: pd.DataFrame, hub_height: float, curve: PowerCurve) -> pd.DataFrame:
    """Ауа райы кестесінен (ws10..sp) NWP белгілерін есептеу."""
    f = pd.DataFrame(index=wx.index)
    for c in ["ws10", "ws80", "ws100", "ws120", "gust10", "t2m", "sp"]:
        f[c] = wx[c] if c in wx else np.nan
    f["gust_factor"] = f["gust10"] / f["ws10"].clip(lower=0.5)
    wd = wx["wd100"] if "wd100" in wx else wx.get("wd10")
    f["u100"], f["v100"] = wind_components(f["ws100"], wd)
    rad = np.deg2rad(wd.astype(float))
    f["wd_sin"], f["wd_cos"] = np.sin(rad), np.cos(rad)
    f["rho"] = air_density(f["sp"], f["t2m"])
    f["alpha"] = shear_exponent(f["ws10"], 10.0, f["ws100"], 100.0)
    f["ws_hub"] = hub_speed(f["ws100"], 100.0, hub_height, f["alpha"])
    f["ws_hub_rho"] = density_corrected_speed(f["ws_hub"], f["rho"].fillna(1.225))
    f["p_phys"] = curve(f["ws_hub_rho"].fillna(0))
    f.loc[f["ws_hub_rho"].isna(), "p_phys"] = np.nan
    f["ws_hub_d1"] = f["ws_hub"].diff()
    f["ws_hub_std3"] = f["ws_hub"].rolling(3, center=True, min_periods=2).std()
    f["ws_hub_prev"] = f["ws_hub"].shift(1)
    f["ws_hub_next"] = f["ws_hub"].shift(-1)
    hour = f.index.hour + f.index.minute / 60
    doy = f.index.dayofyear
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return f


def nwp_dataset(power_norm: pd.Series, feats: pd.DataFrame) -> pd.DataFrame:
    """NWP режимінің оқыту кестесі: жарамды уақыт бойынша белгілер + мақсат y."""
    df = feats.copy()
    df["y"] = power_norm.reindex(df.index)
    return df


def short_dataset(power_norm: pd.Series, feats: pd.DataFrame, horizons=SHORT_HORIZONS) -> pd.DataFrame:
    """Өте қысқа мерзімді режим: әр (issue_time t0, горизонт h) жұбы бір жол."""
    p = power_norm.reindex(feats.index)
    base = pd.DataFrame(index=feats.index)
    base["p_now"] = p
    base["p_lag1"] = p.shift(1)
    base["p_lag2"] = p.shift(2)
    base["p_mean3"] = p.rolling(3, min_periods=2).mean()
    base["p_phys_now"] = feats["p_phys"]
    base["resid_now"] = base["p_now"] - base["p_phys_now"]
    rows = []
    for h in horizons:
        valid = feats.shift(-h)  # t0+h уақытындағы ауа райы
        part = valid.copy()
        part[base.columns] = base
        part["h"] = h
        part["y"] = p.shift(-h)
        part["issue_time"] = part.index
        part["valid_time"] = part.index + pd.Timedelta(hours=h)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out


def persistence_frame(power_norm: pd.Series, horizons) -> pd.DataFrame:
    """Persistence үшін: әр жарамды уақыт t және горизонт h -> p_now = P(t−h), y = P(t)."""
    p = power_norm.asfreq("1h")
    parts = []
    for h in horizons:
        parts.append(pd.DataFrame({"valid_time": p.index, "h": h, "p_now": p.shift(h).to_numpy(), "y": p.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def select_by_lead(sources: dict[int, pd.DataFrame], lead_hours: int) -> pd.DataFrame:
    """Алдын ала болжау уақытына сәйкес NWP дереккөзін таңдау.

    sources: {0: ағымдағы/қысқа болжам, 1: previous_day1, 2: previous_day2, 3: previous_day3}
    lead 0–23 -> 0, 24–47 -> 1, 48–71 -> 2, 72 -> 3 (бар болса, әйтпесе ең жақын).
    """
    day = min(lead_hours // 24, max(sources))
    while day not in sources and day > 0:
        day -= 1
    return sources[day]
