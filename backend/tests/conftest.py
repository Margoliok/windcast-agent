import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SCHEDULER_ENABLED", "false")


def _greenbyte_turbine_csv(index: pd.DatetimeIndex, ws: np.ndarray, power: np.ndarray, pitch: np.ndarray) -> str:
    """Kelmarsh SCADA форматындағы шағын синтетикалық файл (тек парсер мен тазалауды тексеру үшін)."""
    head = [
        "# This file was exported by Greenbyte at 2024-01-01 00:00:00.",
        "#",
        "# Turbine: Kelmarsh 1",
        "# Turbine type: MM92",
        "# Time zone: UTC",
        "# Time interval: test",
        "#",
        "# Data that is missing or is erroneous has been marked with the value \"NaN\"",
        "#",
        "# Date and time,Wind speed (m/s),Wind direction (°),Power (kW),Blade angle (pitch position) A (°),Nacelle ambient temperature (°C)",
    ]
    body = [f"{t:%Y-%m-%d %H:%M:%S},{w:.2f},200,{'NaN' if np.isnan(p) else f'{p:.1f}'},{b:.1f},10"
            for t, w, p, b in zip(index, ws, power, pitch)]
    return "\n".join(head + body) + "\n"


STATUS_CSV = """# Greenbyte status export
#
Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category
2024-01-01 12:00:00,2024-01-01 14:00:00,02:00:00,Stop,1001,Manual stop,,,Requested Shutdown
2024-01-01 15:00:00,2024-01-01 15:10:00,00:10:00,Warning,2002,Something,,,Full Performance
"""


@pytest.fixture
def scada_zip(tmp_path) -> Path:
    idx = pd.date_range("2024-01-01", periods=6 * 24, freq="10min")
    rng = np.random.default_rng(0)
    ws = 8 + 2 * np.sin(np.arange(len(idx)) / 20) + rng.normal(0, 0.3, len(idx))
    power = np.clip(2050 * ((ws - 3.5) / 9) ** 3, 0, 2050)
    pitch = np.zeros(len(idx))
    power[72:84] = 0.0  # 12:00–14:00 тоқтау (Status-та бар)
    power[100:103] = 0.0  # статуссыз тоқтау — жанама ереже ұстауы керек
    power[110] = np.nan
    p = tmp_path / "Kelmarsh_SCADA_2024_test.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("Turbine_Data_Kelmarsh_1_2024-01-01_-_2024-01-02_1.csv", _greenbyte_turbine_csv(idx, ws, power, pitch))
        z.writestr("Status_Kelmarsh_1_2024-01-01_-_2024-01-02_1.csv", STATUS_CSV)
    return p
