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
