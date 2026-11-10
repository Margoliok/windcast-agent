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
