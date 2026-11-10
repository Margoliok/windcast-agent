# Қосымша 1. Негізгі код листингтері

## Листинг Қ1.1 — Open-Meteo-дан деректерді жүктеу (`backend/app/ingest/openmeteo.py`)

```python
"""Open-Meteo API клиенті: жедел болжам, тарихи болжамдар архиві, алдыңғы жүгірістер, ERA5.

Барлық уақыт UTC (naive). Тарихи сұраныстар data/cache/openmeteo ішінде кэштеледі.

Дереккөздер:
  * Forecast API            — https://api.open-meteo.com/v1/forecast (жұмыс режимі)
  * Historical Forecast API — https://historical-forecast-api.open-meteo.com (NWP-дің қысқа
    алдын ала болжамдары тігілген архив; оқыту үшін)
  * Previous Runs API       — https://previous-runs-api.open-meteo.com (1,2,3 тәулік бұрын
    шыққан болжамдар — горизонт бойынша адал тест үшін)
  * Historical Weather API  — https://archive-api.open-meteo.com (ERA5 реанализі)
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from app.core.config import get_settings

# стандартты баған атауы -> Open-Meteo айнымалысы
VARIABLES: dict[str, str] = {
    "ws10": "wind_speed_10m",
    "ws80": "wind_speed_80m",
    "ws100": "wind_speed_100m",
    "ws120": "wind_speed_120m",
    "wd10": "wind_direction_10m",
    "wd100": "wind_direction_100m",
    "gust10": "wind_gusts_10m",
    "t2m": "temperature_2m",
    "sp": "surface_pressure",
}

URLS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "era5": "https://archive-api.open-meteo.com/v1/archive",
}


class OpenMeteoError(RuntimeError):
    pass


def _cache_path(url: str, params: dict):
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:20]
    d = get_settings().cache_dir / "openmeteo"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _get(url: str, params: dict, cache: bool = True, retries: int = 4) -> dict:
    path = _cache_path(url, params)
    if cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    last = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, params=params, timeout=get_settings().openmeteo_timeout)
            if r.status_code == 429:  # минуттық лимит
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                raise OpenMeteoError(data.get("reason", "Open-Meteo error"))
            if cache:
                path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except (httpx.HTTPError, ValueError) as e:  # желілік қате — қайталаймыз
            last = e
            time.sleep(2 * (attempt + 1))
    raise OpenMeteoError(f"Open-Meteo сұранысы сәтсіз: {last}")


def _to_frame(data: dict, suffix: str = "") -> pd.DataFrame:
    h = data["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])}).set_index("time")
    for std, var in VARIABLES.items():
        key = var + suffix
        if key in h:
            df[std] = pd.to_numeric(pd.Series(h[key], index=df.index), errors="coerce")
    df.index.name = "time"
    return df


def _base_params(lat: float, lon: float) -> dict:
    return {"latitude": lat, "longitude": lon, "wind_speed_unit": "ms", "timezone": "GMT"}


def _year_chunks(start: date, end: date):
    cur = start
    while cur <= end:
        nxt = min(date(cur.year, 12, 31), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def _ranged(kind: str, lat, lon, start, end, extra: dict | None = None, suffix: str = "") -> pd.DataFrame:
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    frames = []
    for a, b in _year_chunks(start, end):
        params = _base_params(lat, lon) | {
            "hourly": ",".join(v + suffix for v in VARIABLES.values()),
            "start_date": a.isoformat(),
            "end_date": b.isoformat(),
        } | (extra or {})
        # ағымдағы жылдың соңғы күндерін кэштемейміз (деректер әлі толықтырылуы мүмкін)
        cache = b < (datetime.now(timezone.utc).date() - timedelta(days=7))
        frames.append(_to_frame(_get(URLS[kind], params, cache=cache), suffix))
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def fetch_historical_forecast(lat, lon, start, end, model: str | None = None) -> pd.DataFrame:
    """NWP тарихи болжамдары (қысқа алдын ала болжау уақыты, ~0–24 сағ)."""
    model = model or get_settings().openmeteo_model
    return _ranged("historical_forecast", lat, lon, start, end, {"models": model})


def fetch_era5(lat, lon, start, end) -> pd.DataFrame:
    """ERA5 реанализі — «мінсіз болжам» жағдайы (жоғарғы шек)."""
    return _ranged("era5", lat, lon, start, end, {"models": "era5"})


def fetch_previous_runs(lat, lon, start, end, day: int, model: str | None = None) -> pd.DataFrame:
    """`day` тәулік бұрын шыққан жүгірістің болжамы (day=1 -> алдын ала болжау 24–47 сағ)."""
    model = model or get_settings().openmeteo_model
    suffix = f"_previous_day{day}" if day > 0 else ""
    return _ranged("previous_runs", lat, lon, start, end, {"models": model}, suffix=suffix)


def fetch_live_forecast(lat, lon, hours: int = 72, past_hours: int = 48) -> pd.DataFrame:
    """Жұмыс режиміндегі болжам. Нәтижеде өткен `past_hours` сағат та бар (лагтар үшін)."""
    params = _base_params(lat, lon) | {
        "hourly": ",".join(VARIABLES.values()),
        "forecast_hours": hours + 1,
        "past_hours": past_hours,
    }
    df = _to_frame(_get(URLS["forecast"], params, cache=False))
    df.attrs["issue_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    return df

```

## Листинг Қ1.2 — Физикалық белгілер: ауа тығыздығы, ығысу, қуат қисығы (`backend/app/features/physics.py`)

```python
"""Жел генерациясының физикасы: ауа тығыздығы, желдің ығысуы, қуат қисығы.

Формулалар (дипломдағы нөмірлеу):
  (1) P = 1/2 · ρ · A · Cp · v³
  (6) ρ = p / (R_d · T)
  (7) v_hub = v_ref · (h_hub / h_ref)^α
  (8) u = -v·sin φ,  w = -v·cos φ
  (10) P̂ = PC( v_hub · (ρ/ρ0)^(1/3) )
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

R_D = 287.05  # құрғақ ауаның газ тұрақтысы, Дж/(кг·К)
RHO_0 = 1.225  # стандартты ауа тығыздығы, кг/м³
BETZ_LIMIT = 16 / 27


def air_density(pressure_hpa, temperature_c):
    """(6) ρ = p / (R_d·T). Қысым гПа, температура °C."""
    p = np.asarray(pressure_hpa, dtype=float) * 100.0
    t = np.asarray(temperature_c, dtype=float) + 273.15
    return p / (R_D * t)


def shear_exponent(v_low, h_low, v_high, h_high, clip=(0.0, 0.6)):
    """Екі биіктіктегі жылдамдық бойынша ығысу көрсеткіші α = ln(v2/v1)/ln(h2/h1)."""
    v1 = np.maximum(np.asarray(v_low, dtype=float), 0.1)
    v2 = np.maximum(np.asarray(v_high, dtype=float), 0.1)
    alpha = np.log(v2 / v1) / np.log(h_high / h_low)
    return np.clip(alpha, *clip)


def hub_speed(v_ref, h_ref, h_hub, alpha):
    """(7) Жылдамдықты ступица биіктігіне келтіру."""
    return np.asarray(v_ref, dtype=float) * (h_hub / h_ref) ** np.asarray(alpha, dtype=float)


def wind_components(speed, direction_deg):
    """(8) Жел векторының компоненттері (метеорологиялық бағыт: жел қайдан соғады)."""
    phi = np.deg2rad(np.asarray(direction_deg, dtype=float))
    v = np.asarray(speed, dtype=float)
    return -v * np.sin(phi), -v * np.cos(phi)


def density_corrected_speed(speed, rho, rho0=RHO_0):
    """IEC 61400-12-1 бойынша тығыздыққа түзетілген жылдамдық."""
    return np.asarray(speed, dtype=float) * (np.asarray(rho, dtype=float) / rho0) ** (1.0 / 3.0)


def theoretical_power_w(rho, rotor_diameter_m, cp, speed):
    """(1) Жел ағынынан алынатын қуат, Вт."""
    area = np.pi * (rotor_diameter_m / 2) ** 2
    return 0.5 * np.asarray(rho) * area * cp * np.asarray(speed) ** 3


@dataclass
class PowerCurve:
    """Нормаланған қуат қисығы: жылдамдық (м/с) -> P/P_inst (0..1).

    Деректерден эмпирикалық түрде құрылады (IEC бин әдісі, бин бойынша медиана).
    """

    speeds: np.ndarray
    power: np.ndarray
    source: str = "empirical"

    def __call__(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        out = np.interp(v, self.speeds, self.power, left=0.0, right=self.power[-1])
        return np.clip(out, 0.0, 1.0)

    @classmethod
    def fit(cls, speed, power_norm, bin_width=0.5, v_max=25.0, min_count=20, source="empirical") -> "PowerCurve":
        df = pd.DataFrame({"v": np.asarray(speed, float), "p": np.asarray(power_norm, float)}).dropna()
        df = df[(df.v >= 0) & (df.v <= v_max)]
        edges = np.arange(0, v_max + bin_width, bin_width)
        df["bin"] = pd.cut(df.v, edges, labels=False, include_lowest=True)
        g = df.groupby("bin").agg(v=("v", "mean"), p=("p", "median"), n=("p", "size"))
        g = g[g.n >= min_count]
        if len(g) < 5:
            raise ValueError("Қуат қисығын құруға деректер жеткіліксіз")
        speeds = np.concatenate([[0.0], g.v.to_numpy()])
        power = np.concatenate([[0.0], np.clip(g.p.to_numpy(), 0, 1)])
        power = np.maximum.accumulate(power)  # монотонды етеміз
        return cls(speeds=speeds, power=power, source=source)

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"speeds": self.speeds.tolist(), "power": self.power.tolist(), "source": self.source}),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> "PowerCurve":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["speeds"]), np.array(d["power"]), d.get("source", "empirical"))

```

## Листинг Қ1.3 — Модель белгілерін қалыптастыру (`backend/app/features/build.py`)

```python
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

```

## Листинг Қ1.4 — LightGBM кванттильдік регрессиясы және конформдық түзету (`backend/app/models/lgbm.py`)

```python
"""M2 — LightGBM кванттильдік регрессиясы.

Әр τ ∈ {0.1, …, 0.9} үшін бөлек модель (5)-пинбол функциясын минимизациялайды
(GEFCom2014 жеңімпаздарының тәсілі). Кейін кванттильдер реттеледі және [0,1]-ге қиылады.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.models.base import Forecaster, finalize_quantiles

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


class LGBMQuantileModel(Forecaster):
    name = "lgbm"
    title = {"kk": "LightGBM кванттильдік (M2)", "ru": "LightGBM квантильный (M2)"}

    def __init__(self, features: list[str], params: dict | None = None, seed: int = 42):
        self.features = list(features)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.offsets: dict[float, float] = {}  # конформдық түзетулер (CQR)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMQuantileModel":
        m = np.isfinite(np.asarray(y, float))
        Xf, yf = X.loc[m, self.features], np.asarray(y, float)[m]
        for tau in self.taus:
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=tau, random_state=self.seed, **self.params)
            mdl.fit(Xf, yf)
            self.models[tau] = mdl
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.features]
        q = np.column_stack([self.models[t].predict(Xf) + self.offsets.get(t, 0.0) for t in self.taus])
        return finalize_quantiles(q, self.taus, X.index)

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> dict[float, float]:
        """Конформдық кванттильдік регрессия (Romano et al., 2019) идеясы бойынша түзету:
        валидацияда әр τ үшін δ_τ = Quantile_τ(y − q̂_τ), сонда P(y ≤ q̂_τ + δ_τ) ≈ τ.
        P50 түзетілмейді (нүктелік болжам өзгермеуі үшін)."""
        m = np.isfinite(np.asarray(y, float))
        Xf, yf = X.loc[m, self.features], np.asarray(y, float)[m]
        self.offsets = {}
        for t in self.taus:
            if t == 0.5:
                continue
            self.offsets[t] = float(np.quantile(yf - self.models[t].predict(Xf), t))
        return self.offsets

    @property
    def median_model(self) -> lgb.LGBMRegressor:
        return self.models[0.5]

```

## Листинг Қ1.5 — Тексеру шлюзі (verification gate) (`backend/app/agent/gate.py`)

```python
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

```

## Листинг Қ1.6 — ЖИ-агенттің құралдары (`backend/app/agent/tools.py`)

```python
"""ЖИ-агенттің 10 құралы (Кесте 11). Әр құрал services/* функцияларын шақырады.

Құрал нәтижесі екі нұсқада сақталады:
  * full    — толық нәтиже (интерфейс графиктері және faithfulness тексерісі үшін);
  * compact — LLM-ге берілетін ықшам нұсқа (токен үнемдеу; сандар full-дың ішкі жиыны).
Қате болса — {"error": "..."}: агент оны ойдан сан шығармай, ашық хабарлауы керек.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from app.services import forecast_service as fs
from app.services import history_service as hs
from app.services.report import generate_report
from app.services.schedule_export import build_dayahead_schedule

FarmId = Literal["kelmarsh", "ereymentau"]
ModelName = Literal["ensemble", "lgbm", "physical"]
LT = "Жергілікті уақыт, формат 'YYYY-MM-DD HH:MM'"


class WeatherArgs(BaseModel):
    farm_id: FarmId = Field(description="Станция идентификаторы")
    hours: int = Field(72, ge=1, le=168, description="Болжам горизонты, сағат")
    low_wind_threshold: float | None = Field(None, description="Әлсіз жел шегі, м/с (ступица биіктігінде). Берілсе — сол шектен төмен кезеңдер ізделеді")
    min_duration_h: int = Field(6, ge=1, le=72, description="Әлсіз жел кезеңінің ең аз ұзақтығы, сағат")


class HistoryArgs(BaseModel):
    farm_id: FarmId
    start_local: str | None = Field(None, description=LT + " (кезең басы)")
    end_local: str | None = Field(None, description=LT + " (кезең соңы, кірмейді)")


class PredictArgs(BaseModel):
    farm_id: FarmId
    hours: int = Field(72, ge=1, le=168, description="Болжам горизонты, сағат")
    model: ModelName | None = Field(None, description="Модель; бос болса — ең үздік (ансамбль)")
    start_local: str | None = Field(None, description=LT + ". Берілсе — энергия мен статистика осы кезең үшін есептеледі")
    end_local: str | None = Field(None, description=LT + " (кірмейді)")


class ExplainArgs(BaseModel):
    farm_id: FarmId
    start_local: str | None = Field(None, description=LT + " — түсіндірілетін кезең басы")
    end_local: str | None = Field(None, description=LT + " — кезең соңы")
    reference_start_local: str | None = Field(None, description="Салыстыру кезеңінің басы (мысалы, күндізгі сағаттар)")
    reference_end_local: str | None = Field(None, description="Салыстыру кезеңінің соңы")


class BacktestArgs(BaseModel):
    farm_id: FarmId
    model: Literal["ensemble", "lgbm", "physical", "persistence", "lgbm_short", "lstm", "tft", "chronos"] = "ensemble"
    start_local: str | None = None
    end_local: str | None = None
    horizon: int = Field(24, description="Горизонт, сағат: 1, 6, 24, 48 немесе 72")


class CompareArgs(BaseModel):
    farm_id: FarmId
    start_local: str | None = None
    end_local: str | None = None
    horizon: int = Field(24, description="Горизонт, сағат: 1, 6, 24, 48 немесе 72")


class WhatIfArgs(BaseModel):
    farm_id: FarmId
    delta_wind_pct: float = Field(description="Жел жылдамдығының болжамнан өзгерісі, % (мысалы 15 немесе -20)")
    hours: int = Field(24, ge=1, le=168)
    start_local: str | None = None
    end_local: str | None = None


class DowntimeArgs(BaseModel):
    farm_id: FarmId
    start_local: str | None = None
    end_local: str | None = None
    min_duration_h: int = Field(2, ge=1, le=48)


class ScheduleArgs(BaseModel):
    farm_id: FarmId
    date_local: str | None = Field(None, description="Күн 'YYYY-MM-DD' (жергілікті); бос болса — ертеңгі күн")


class ReportArgs(BaseModel):
    farm_id: FarmId
    hours: int = Field(72, ge=6, le=168)


def _series_compact(rows: list[dict], keys: tuple[str, ...]) -> list[list]:
    return [[r.get(k) for k in keys] for r in rows]


def _compact(name: str, out: dict) -> dict:
    """LLM-ге берілетін ықшам нұсқа."""
    if "error" in out:
        return out
    o = {k: v for k, v in out.items() if k not in ("series", "schedule", "items")}
    if name == "predict_power" and "series" in out:
        o["series_columns"] = ["time_local", "p10_mw", "p50_mw", "p90_mw", "ws_hub_ms"]
        o["series"] = _series_compact(out["series"], ("time_local", "p10", "p50", "p90", "ws_hub"))
    elif name == "get_weather_forecast":
        o["series_columns"] = ["time_local", "ws_hub_ms", "ws100_ms", "wd100_deg", "gust10_ms", "t2m_c"]
        o["series"] = _series_compact(out["series"], ("time_local", "ws_hub", "ws100", "wd100", "gust10", "t2m"))
    elif name == "build_dayahead_schedule":
        o["schedule_columns"] = ["hour_local", "p10_mw", "p50_mw", "p90_mw", "ws_hub_ms"]
        o["schedule"] = _series_compact(out["schedule"], ("hour_local", "p10", "p50", "p90", "ws_hub"))
    elif name == "get_history" and "series" in out:
        o["note"] = "Сағаттық қатар интерфейсте графикпен көрсетіледі; мұнда тек қорытынды."
    elif name == "what_if":
        o.pop("series", None)
    return o


def _wrap(name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
    def run(**kwargs) -> dict:
        t0 = time.time()
        try:
            out = fn(**kwargs)
        except (fs.ServiceError, ValueError, FileNotFoundError) as e:
            out = {"error": str(e)}
        except Exception as e:  # күтпеген қате — агент ашық хабарлайды
            out = {"error": f"Ішкі қате: {type(e).__name__}: {e}"}
        out = {"tool": name, **out}
        out["_elapsed_s"] = round(time.time() - t0, 2)
        return out

    return run


TOOLS: dict[str, dict[str, Any]] = {
    "get_weather_forecast": {
        "fn": fs.weather_forecast, "args": WeatherArgs,
        "desc": "Open-Meteo ауа райы болжамы (жел 10/100 м және ступица биіктігінде, бағыт, екпін, температура). "
                "low_wind_threshold берілсе — әлсіз жел кезеңдерін табады (ТҚК жоспарлау үшін).",
    },
    "get_history": {
        "fn": lambda **k: hs.get_history(**k), "args": HistoryArgs,
        "desc": "Дерекқордан тарихи өндіріс (SCADA) және ERA5 жел деректері, кезең қорытындысы. Тек Kelmarsh.",
    },
    "predict_power": {
        "fn": fs.predict_power, "args": PredictArgs,
        "desc": "ЖЭС қуатының болжамы P10/P50/P90 (МВт), сағаттық. start_local/end_local берілсе — осы кезеңдегі "
                "энергия (МВт·сағ), орташа/максимум/минимум қуат дайын есептеледі.",
    },
    "explain_forecast": {
        "fn": fs.explain_forecast, "args": ExplainArgs,
        "desc": "SHAP бойынша болжамға әсер еткен негізгі факторлар (МВт). Салыстыру кезеңі берілсе — өзгерістің себептері "
                "(мысалы, «кешкі төмендеу неге»).",
    },
    "backtest": {
        "fn": hs.backtest, "args": BacktestArgs,
        "desc": "Тест кезеңіндегі модель сапасының метрикалары: MAE, RMSE, nMAE, skill score, CRPS, PICP. Тек Kelmarsh.",
    },
    "compare_models": {
        "fn": hs.compare_models, "args": CompareArgs,
        "desc": "Модельдерді (Persistence, физикалық, LightGBM, ансамбль, т.б.) бір кезеңде салыстыру кестесі. Тек Kelmarsh.",
    },
    "what_if": {
        "fn": fs.what_if, "args": WhatIfArgs,
        "desc": "Сценарий: жел болжамнан delta_wind_pct %-ға күшті/әлсіз болса, өндіріс қалай өзгереді.",
    },
    "detect_downtime": {
        "fn": hs.detect_downtime, "args": DowntimeArgs,
        "desc": "Жел жеткілікті болса да турбиналар жұмыс істемеген (тоқтау/шектеу) кезеңдер және жоғалған энергия. Тек Kelmarsh.",
    },
    "build_dayahead_schedule": {
        "fn": build_dayahead_schedule, "args": ScheduleArgs,
        "desc": "Келесі тәулікке (немесе көрсетілген күнге) сағаттық өндіріс кестесі P10/P50/P90 және CSV/XLSX файлдары.",
    },
    "generate_report": {
        "fn": generate_report, "args": ReportArgs,
        "desc": "Графиктері бар есеп (Markdown және PDF): болжам қорытындысы, әлсіз жел кезеңдері, модель сапасы.",
    },
}


def run_tool(name: str, args: dict, lang: str = "kk") -> dict:
    spec = TOOLS.get(name)
    if spec is None:
        return {"tool": name, "error": f"Белгісіз құрал: {name}"}
    try:
        parsed = spec["args"](**args).model_dump()
    except Exception as e:
        return {"tool": name, "error": f"Аргументтер қате: {e}"}
    if name in ("explain_forecast", "build_dayahead_schedule", "generate_report"):
        parsed["lang"] = lang
    return _wrap(name, spec["fn"])(**parsed)


def compact(name: str, out: dict) -> dict:
    return _compact(name, out)


def langchain_tools():
    """LangChain StructuredTool тізімі (LLM-ге схемаларды беру үшін)."""
    from langchain_core.tools import StructuredTool

    tools = []
    for name, spec in TOOLS.items():
        tools.append(StructuredTool.from_function(
            func=lambda **kw: None, name=name, description=spec["desc"], args_schema=spec["args"]))
    return tools

```

## Листинг Қ1.7 — ЖИ-агенттің LangGraph графы (`backend/app/agent/graph.py`)

```python
"""ЖИ-агенттің LangGraph күйлер графы (Сурет 13).

    START → agent ──(tool_calls)──→ tools ──→ agent
              │
              └──(жауап жобасы)──→ verify ──(өтті)──→ END
                                     │
                                     ├──(өтпеді, қайталау < N)──→ agent (шлюз ескертуімен)
                                     └──(өтпеді, лимит)──→ safe_fail → END

ReAct циклі: LLM ойлайды → құрал шақырады → нәтижені бақылайды → ... → жауап.
Тексеру шлюзі (verify) — детерминистік код: LLM-нің жауабындағы әр сан құрал нәтижесімен сәйкес келуі тиіс.
"""
from __future__ import annotations

import json
import operator
import time
import uuid
from typing import Annotated, Any, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent import prompts
from app.agent.gate import verify_answer
from app.agent.llm import LLMNotConfigured, get_chat_model, usage_tokens
from app.agent.tools import compact, langchain_tools, run_tool
from app.core.config import get_settings
from app.core.farms import get_farm


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    lang: str
    farm_id: str
    tool_outputs: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    retries: int
    rounds: int
    tokens: int
    gate: dict
    final: str
    status: str  # ok | safe_fail | error


def _system_prompt(lang: str, farm_id: str) -> str:
    from app.services.forecast_service import now_utc, to_local

    farm = get_farm(farm_id)
    now = to_local(now_utc(), farm)
    try:
        from app.services.history_service import data_range

        r = data_range("kelmarsh")
        hist = f"{r['start_local']} – {r['end_local']}"
    except Exception:
        hist = "—"
    return prompts.SYSTEM[lang].format(
        now_local=now.strftime("%Y-%m-%d %H:%M (%A)"), tomorrow=(now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        farm_id=farm.id, farm_name=farm.name, capacity=farm.capacity_mw, farms=prompts.farms_text(), history_range=hist,
    )


def build_graph(llm=None):
    s = get_settings()
    model = llm or get_chat_model()
    bound = model.bind_tools(langchain_tools())

    def agent(state: AgentState) -> dict:
        t0 = time.time()
        msg = bound.invoke(state["messages"])
        calls = [{"name": c["name"], "args": c["args"]} for c in (getattr(msg, "tool_calls", None) or [])]
        ev = {"step": "llm", "t": round(time.time() - t0, 2), "tool_calls": calls,
              "draft": None if calls else (msg.content or "")[:4000]}
        return {"messages": [msg], "trace": [ev], "tokens": state.get("tokens", 0) + usage_tokens(msg),
                "rounds": state.get("rounds", 0) + (1 if calls else 0)}

    def tools(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        msgs, outs, trace = [], [], []
        for call in last.tool_calls:
            args = dict(call["args"])
            args.setdefault("farm_id", state["farm_id"])
            out = run_tool(call["name"], args, lang=state["lang"])
            outs.append(out)
            trace.append({"step": "tool", "name": call["name"], "args": args, "t": out.get("_elapsed_s"),
                          "error": out.get("error"), "gate": out.get("gate"),
                          "summary": out.get("summary") or out.get("metrics")})
            msgs.append(ToolMessage(content=json.dumps(compact(call["name"], out), ensure_ascii=False, default=str),
                                    tool_call_id=call["id"], name=call["name"]))
        return {"messages": msgs, "tool_outputs": outs, "trace": trace}

    def route_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("rounds", 0) <= s.agent_max_tool_rounds:
            return "tools"
        return "verify"

    def verify(state: AgentState) -> dict:
        draft = state["messages"][-1].content or ""
        g = verify_answer(draft, state.get("tool_outputs", []), state["question"]).to_dict()
        ev = {"step": "gate", "passed": g["passed"], "checks": g["checks"], "retry": state.get("retries", 0)}
        upd: dict[str, Any] = {"gate": g, "trace": [ev]}
        if g["passed"]:
            upd.update(final=draft, status="ok")
        else:
            nums = next((c.get("unsupported") for c in g["checks"] if c["check"] == "faithfulness"), []) or []
            upd["retries"] = state.get("retries", 0) + 1
            if upd["retries"] <= s.agent_max_retries:
                failed = [c["check"] for c in g["checks"] if not c["passed"]]
                txt = prompts.GATE_FEEDBACK[state["lang"]].format(nums=", ".join(map(str, nums)) or ", ".join(failed))
                upd["messages"] = [HumanMessage(content=txt)]
        return upd

    def route_verify(state: AgentState) -> str:
        if state.get("status") == "ok":
            return END
        return "agent" if state.get("retries", 0) <= s.agent_max_retries else "safe_fail"

    def safe_fail(state: AgentState) -> dict:
        g = state.get("gate", {})
        nums = next((c.get("unsupported") for c in g.get("checks", []) if c["check"] == "faithfulness"), []) or []
        # расталмаған сандардың өзі пайдаланушыға көрсетілмейді (тек саны) — олар трассада қалады
        txt = prompts.SAFE_FAIL[state["lang"]].format(n=len(nums) or 1)
        return {"final": txt, "status": "safe_fail", "trace": [{"step": "safe_fail"}]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("verify", verify)
    g.add_node("safe_fail", safe_fail)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_agent, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", route_verify, {END: END, "agent": "agent", "safe_fail": "safe_fail"})
    g.add_edge("safe_fail", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def initial_state(question: str, lang: str, farm_id: str, history: list[dict] | None = None) -> AgentState:
    lang = lang if lang in prompts.SYSTEM else "kk"
    msgs: list[BaseMessage] = [SystemMessage(content=_system_prompt(lang, farm_id))]
    for h in (history or [])[-6:]:  # қысқа диалог контексі (тек мәтін)
        if h.get("role") == "user":
            msgs.append(HumanMessage(content=h.get("content", "")))
        elif h.get("role") == "assistant":
            msgs.append(AIMessage(content=h.get("content", "")))
    msgs.append(HumanMessage(content=question))
    return {"messages": msgs, "question": question, "lang": lang, "farm_id": farm_id, "tool_outputs": [],
            "trace": [], "retries": 0, "rounds": 0, "tokens": 0}


def _result(state: dict, t0: float, session_id: str) -> dict:
    return {
        "session_id": session_id,
        "answer": state.get("final", ""),
        "status": state.get("status", "error"),
        "gate": state.get("gate"),
        "trace": state.get("trace", []),
        "artifacts": [_artifact(o) for o in state.get("tool_outputs", []) if "error" not in o],
        "tools_called": [o.get("tool") for o in state.get("tool_outputs", [])],
        "latency_s": round(time.time() - t0, 2),
        "tokens": state.get("tokens", 0),
    }


def _artifact(out: dict) -> dict:
    """Интерфейс үшін: құрал нәтижесінен график/кесте/файл."""
    keep = ("tool", "farm", "summary", "series", "schedule", "files", "table", "metrics", "periods",
            "low_wind_windows", "factors", "groups", "baseline", "scenario", "unit", "capacity_mw", "model")
    return {k: out[k] for k in keep if k in out}


def ask(question: str, lang: str = "kk", farm_id: str = "kelmarsh", session_id: str | None = None,
        history: list[dict] | None = None, log: bool = True) -> dict:
    session_id = session_id or uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        state = get_graph().invoke(initial_state(question, lang, farm_id, history), {"recursion_limit": 40})
    except LLMNotConfigured as e:
        state = {"final": prompts.LLM_ERROR[lang if lang in prompts.LLM_ERROR else "kk"].format(err=e), "status": "error", "trace": []}
    except Exception as e:  # API қатесі және т.б.
        state = {"final": prompts.LLM_ERROR[lang if lang in prompts.LLM_ERROR else "kk"].format(err=f"{type(e).__name__}: {e}"),
                 "status": "error", "trace": []}
    res = _result(state, t0, session_id)
    if log:
        log_interaction(res, question, lang, "agent")
    return res


def stream(question: str, lang: str = "kk", farm_id: str = "kelmarsh", session_id: str | None = None,
           history: list[dict] | None = None):
    """SSE үшін: әр қадамнан кейін оқиға шығарады, соңында толық нәтиже."""
    session_id = session_id or uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        graph = get_graph()
    except LLMNotConfigured as e:
        res = _result({"final": prompts.LLM_ERROR[lang].format(err=e), "status": "error"}, t0, session_id)
        yield {"event": "final", "data": res}
        return
    acc: dict[str, Any] = dict(initial_state(question, lang, farm_id, history))
    try:
        for upd in graph.stream(acc, {"recursion_limit": 40}, stream_mode="updates"):
            for node, delta in upd.items():
                if not delta:
                    continue
                for ev in delta.get("trace", []):
                    yield {"event": "trace", "data": {"node": node, **_jsonable(ev)}}
                for k, v in delta.items():
                    if k in ("trace", "tool_outputs"):
                        acc[k] = acc.get(k, []) + v
                    elif k != "messages":
                        acc[k] = v
    except Exception as e:
        acc.update(final=prompts.LLM_ERROR[lang].format(err=f"{type(e).__name__}: {e}"), status="error")
    res = _result(acc, t0, session_id)
    log_interaction(res, question, lang, "agent")
    yield {"event": "final", "data": res}


def _jsonable(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def log_interaction(res: dict, question: str, lang: str, system: str) -> None:
    try:
        from app.core.db import session
        from app.db.models import AgentLog

        faith = None
        for c in (res.get("gate") or {}).get("checks", []):
            if c.get("check") == "faithfulness":
                faith = c.get("score")
        with session() as s:
            s.add(AgentLog(session_id=res["session_id"], lang=lang, question=question, tools_called=res.get("tools_called"),
                           answer=res.get("answer", ""), gate_result=_jsonable(res.get("gate")), faithfulness=faith,
                           latency_s=res.get("latency_s"), tokens=res.get("tokens"), system=system))
    except Exception:
        pass


def export_graph_png(path) -> bool:
    """Сурет 13 үшін графтың Mermaid сипаттамасы (.mmd). PNG-ні figures.py өзі салады
    (сыртқы рендер-қызметтерге сұраныс жіберілмейді)."""
    from pathlib import Path

    class _Dummy:
        def bind_tools(self, tools):
            return self

    g = build_graph(llm=_Dummy())
    Path(path).with_suffix(".mmd").write_text(g.get_graph().draw_mermaid(), encoding="utf-8")
    return True

```

## Листинг Қ1.8 — Гиперпараметрлер (YAML)

```yaml
# Гиперпараметрлер конфигурациясы (Optuna, rolling-origin CV)
mode: "grid_meter"
splits:
  train: ["2017-01-01", "2018-06-30 23:00"]
  val: ["2018-07-01", "2018-12-31 23:00"]
  test: ["2019-01-01", "2019-09-30 23:00"]
curve_source: "nwp_hub_speed"
lgbm_params:
  num_leaves: 53
  learning_rate: 0.025070353438149492
  n_estimators: 1100
  min_child_samples: 180
  colsample_bytree: 0.8461446310098963
  reg_lambda: 8.11016952093725
conformal_offsets:
  0.1: -0.026801970959640334
  0.2: -0.011986228692984335
  0.3: -0.00652893136371517
  0.4: -0.004100770740380725
  0.6: 0.0002507760036508151
  0.7: 0.002475558198106794
  0.8: 0.008171557043413094
  0.9: 0.01868761269175282
best_model: "ensemble"
```
