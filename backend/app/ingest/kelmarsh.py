"""Kelmarsh ЖЭС ашық деректерін (Zenodo 16807551, CC-BY-4.0) жүктеу және оқу.

Greenbyte экспорт форматы: бірнеше '#' түсініктеме жолы, кейін '# Date and time,...' тақырыбы.
SCADA архивінде әр турбинаға екі файл бар:
  Turbine_Data_Kelmarsh_<N>_<from>_-_<to>_<id>.csv  — 10 минуттық сигналдар
  Status_Kelmarsh_<N>_<from>_-_<to>_<id>.csv        — оқиғалар (тоқтау, ескерту ...)
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ZENODO_RECORD = "16807551"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
SMALL_FILES = ["Kelmarsh_WT_static.csv", "Kelmarsh_WT_dataSignalMapping.csv", "Kelmarsh_Grid_Meter_4458.zip"]

# SCADA-дан оқылатын бағандар (тақырып басы бойынша іздейміз)
SCADA_COLUMNS = {
    "ws": "Wind speed (m/s)",
    "wd": "Wind direction (°)",
    "power_kw": "Power (kW)",
    "pitch": "Blade angle (pitch position) A (°)",
    "setpoint_kw": "Turbine Power setpoint (kW)",
    "t_amb": "Nacelle ambient temperature (°C)",
}


# --------------------------------------------------------------------------- download
def list_zenodo_files() -> list[dict]:
    r = httpx.get(ZENODO_API, timeout=60)
    r.raise_for_status()
    return [
        {"key": f["key"], "size": f["size"], "checksum": f.get("checksum", ""), "url": f["links"]["self"]}
        for f in r.json()["files"]
    ]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(info: dict, dest_dir: Path, progress=print) -> Path:
    """Докачкамен (HTTP Range) жүктеу және MD5 тексеру."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / info["key"]
    expected = info["checksum"].split(":")[-1] if info.get("checksum") else ""
    if dest.exists() and dest.stat().st_size == info["size"]:
        if not expected or _md5(dest) == expected:
            progress(f"  ✓ {info['key']} (бар)")
            return dest
    part = dest.with_suffix(dest.suffix + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with httpx.stream("GET", info["url"], headers=headers, timeout=120, follow_redirects=True) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()
        if r.status_code == 200:
            pos = 0
        with open(part, "ab" if pos else "wb") as fh:
            done = pos
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if done % (50 << 20) < (1 << 20):
                    progress(f"  {info['key']}: {done / 1e6:.0f}/{info['size'] / 1e6:.0f} MB")
    if expected and _md5(part) != expected:
        part.unlink()
        raise IOError(f"MD5 сәйкес емес: {info['key']}")
    part.replace(dest)
    progress(f"  ✓ {info['key']}")
    return dest


# --------------------------------------------------------------------------- parsing
def _read_greenbyte(raw: bytes, wanted: dict[str, str] | None = None) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines[:40]) if ln.startswith("# Date and time"))
    header = pd.read_csv(io.StringIO(lines[hdr].lstrip("# ")), nrows=0).columns.tolist()
    usecols = [header[0]]
    rename = {header[0]: "time"}
    if wanted:
        for std, prefix in wanted.items():
            col = next((c for c in header if c == prefix), None) or next(
                (c for c in header if c.startswith(prefix.split(" (")[0] + " (")), None
            )
            if col:
                usecols.append(col)
                rename[col] = std
    df = pd.read_csv(
        io.StringIO("\n".join(lines[hdr + 1 :])), names=header, usecols=usecols, na_values=["NaN"], low_memory=False
    ).rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("time").sort_index()


def read_grid_meter(zip_path: Path) -> pd.DataFrame:
    """Станция есептегіші: 10 мин энергия (кВт·сағ) -> сағаттық орташа қуат (МВт)."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.startswith("Device_Data"))
        df = _read_greenbyte(z.read(name), {"energy_kwh": "Grid Meter Energy Export (kWh)"})
    e = df["energy_kwh"]
    hourly_mwh = hourly_centered(e, "sum", min_count=6) / 1000.0  # 6 интервал болса ғана
    out = pd.DataFrame({"power_mw": hourly_mwh})
    out["n_intervals"] = hourly_centered(e, "count")
    return out


def hourly_centered(s: pd.Series, how: str = "mean", **kw) -> pd.Series:
    """10 минуттық деректерді сағат белгісіне ОРТАЛАНҒАН терезеге агрегациялау: [t−30, t+30) мин.

    Open-Meteo жел жылдамдығы — t сәтіндегі лездік мән, сондықтан сағаттық қуатты дәл
    сол сәтке орталау уақыттық ығысуды жояды (10-мин интервал белгісі — оның басы).
    """
    r = s.resample("1h", offset="30min")
    out = getattr(r, how)(**kw)
    out.index = out.index + pd.Timedelta(minutes=30)
    return out


def read_status(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines[:40]) if ln.startswith("Timestamp start"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), low_memory=False)
    df["Timestamp start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
    df["Timestamp end"] = pd.to_datetime(df["Timestamp end"], errors="coerce")
    return df


_TURB_RE = re.compile(r"Kelmarsh_(\d+)_")


def read_scada_zip(zip_path: Path) -> dict[int, dict[str, pd.DataFrame]]:
    """SCADA жылдық архивін оқу -> {турбина: {"data": df10min, "status": df}}."""
    out: dict[int, dict[str, pd.DataFrame]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            m = _TURB_RE.search(name)
            if not m or not name.endswith(".csv"):
                continue
            t = int(m.group(1))
            base = Path(name).name
            if base.startswith("Turbine_Data"):
                out.setdefault(t, {})["data"] = _read_greenbyte(z.read(name), SCADA_COLUMNS)
            elif base.startswith("Status"):
                out.setdefault(t, {})["status"] = read_status(z.read(name))
    return out


def stop_mask(index: pd.DatetimeIndex, status: pd.DataFrame | None) -> pd.Series:
    """Тоқтау (Stop) оқиғалары жабатын 10 минуттық интервалдар."""
    if status is None or status.empty or len(index) == 0:
        return pd.Series(False, index=index)
    stops = status[status["Status"].astype(str).str.lower().eq("stop")].dropna(subset=["Timestamp start"])
    t = index.values
    starts_iv = (index + pd.Timedelta(minutes=10)).values
    ends = stops["Timestamp end"].fillna(index.max() + pd.Timedelta(minutes=10)).values
    # 10-мин интервал [t, t+10) оқиғамен [s, e) қиылысады <=> t+10 > s және t < e
    lo = pd.Index(starts_iv).searchsorted(stops["Timestamp start"].values, side="right")
    hi = pd.Index(t).searchsorted(ends, side="left")
    diff = np.zeros(len(index) + 1, dtype=np.int64)
    for a, b in zip(lo, hi):
        if b > a:
            diff[a] += 1
            diff[b] -= 1
    return pd.Series(diff.cumsum()[:-1] > 0, index=index)
