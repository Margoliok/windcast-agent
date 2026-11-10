import numpy as np
import pandas as pd
import pytest

from app.evaluation import metrics as M
from app.features.build import persistence_frame, short_dataset
from app.features.physics import (
    PowerCurve,
    air_density,
    density_corrected_speed,
    hub_speed,
    shear_exponent,
    wind_components,
)


def test_air_density_standard_atmosphere():
    assert air_density(1013.25, 15.0) == pytest.approx(1.225, abs=2e-3)


def test_shear_and_hub_speed_roundtrip():
    alpha = shear_exponent(5.0, 10, 5.0 * (100 / 10) ** 0.14, 100)
    assert alpha == pytest.approx(0.14, abs=1e-6)
    assert hub_speed(8.0, 100, 100, 0.2) == pytest.approx(8.0)
    assert hub_speed(8.0, 100, 80, 0.2) < 8.0


def test_wind_components_meteorological_convention():
    u, v = wind_components(10.0, 0.0)  # солтүстіктен соғады -> оңтүстікке қарай ағады
    assert u == pytest.approx(0, abs=1e-9) and v == pytest.approx(-10.0)
    u, v = wind_components(10.0, 270.0)  # батыстан -> шығысқа
    assert u == pytest.approx(10.0) and v == pytest.approx(0, abs=1e-9)


def test_density_correction():
    assert density_corrected_speed(10.0, 1.225) == pytest.approx(10.0)
    assert density_corrected_speed(10.0, 1.3) > 10.0


def test_power_curve_fit_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    v = rng.uniform(0, 25, 20000)
    p = np.clip(((v - 3.5) / 9) ** 3, 0, 1) + rng.normal(0, 0.03, v.size)
    c = PowerCurve.fit(v, p)
    grid = np.linspace(0, 25, 200)
    out = c(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1
    assert c(15.0) == pytest.approx(1.0, abs=0.05)
    assert c(2.0) == pytest.approx(0.0, abs=0.05)


def test_point_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.0, 3.0, 1.0])
    assert M.mae(y, yhat) == pytest.approx(1.0)
    assert M.rmse(y, yhat) == pytest.approx(np.sqrt(5 / 3))
    assert M.nmae(y, yhat, capacity=10) == pytest.approx(10.0)
    assert M.skill_score(1.0, 2.0) == pytest.approx(0.5)
    assert M.bias(y, yhat) == pytest.approx(-1 / 3)


def test_pinball_and_crps():
    y = np.array([1.0])
    assert M.pinball(y, np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert M.pinball(y, np.array([2.0]), 0.9) == pytest.approx(0.1)
    q = np.array([[1.0, 1.0, 1.0]])
    assert M.crps_from_quantiles(y, q, [0.1, 0.5, 0.9]) == pytest.approx(0.0)


def test_picp():
    y = np.array([0.5, 0.5, 0.5, 0.5])
    assert M.picp(y, [0.4, 0.6, 0.0, 0.4], [0.6, 0.7, 0.4, 0.5]) == pytest.approx(50.0)


def test_short_dataset_alignment():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    p = pd.Series(np.arange(48) / 100.0, index=idx)
    feats = pd.DataFrame({"p_phys": 0.1, "ws100": np.arange(48.0)}, index=idx)
    S = short_dataset(p, feats, horizons=(1, 3))
    r = S[(S.h == 3) & (S.issue_time == idx[10])].iloc[0]
    assert r.p_now == pytest.approx(0.10)
    assert r.y == pytest.approx(0.13)  # P(t0+3)
    assert r.ws100 == pytest.approx(13.0)  # жарамды уақыттағы ауа райы
    P = persistence_frame(p, [2])
    row = P[P.valid_time == idx[5]].iloc[0]
    assert row.p_now == pytest.approx(0.03) and row.y == pytest.approx(0.05)
