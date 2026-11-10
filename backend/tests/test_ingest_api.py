import numpy as np
import pytest

from app.ingest.cleaning import clean_turbine, quality_row, station_hourly
from app.ingest.kelmarsh import read_scada_zip


def test_read_and_clean_scada(scada_zip):
    data = read_scada_zip(scada_zip)
    assert set(data) == {1}
    df = data[1]["data"]
    assert {"ws", "power_kw", "pitch"} <= set(df.columns)
    assert len(df) == 144
    c = clean_turbine(df, data[1]["status"])
    assert c.loc["2024-01-01 12:00":"2024-01-01 13:50", "f_stop"].all()  # Status: Stop
    assert not c.loc["2024-01-01 15:00", "f_stop"]  # Warning — тоқтау емес
    assert c.iloc[100:103]["f_zero_at_wind"].all()  # статуссыз тоқтау
    assert c.iloc[110]["f_missing"]
    q = quality_row(1, c)
    assert 0 < q["clean_pct"] < 100


def test_station_hourly_centered_and_scaled(scada_zip):
    data = read_scada_zip(scada_zip)
    c = clean_turbine(data[1]["data"], data[1]["status"])
    h = station_hourly({1: c}, min_frac=1.0)
    assert np.isnan(h.loc["2024-01-01 13:00", "power_mw"])  # тоқтау сағаты алынып тасталды
    assert h.loc["2024-01-01 06:00", "power_mw"] == pytest.approx(
        c.loc["2024-01-01 05:30":"2024-01-01 06:20", "power_kw"].mean() / 1000, rel=1e-6)


def test_tools_validate_args_and_report_errors():
    from app.agent.tools import run_tool

    out = run_tool("predict_power", {"farm_id": "unknown"})
    assert "error" in out
    out = run_tool("get_history", {"farm_id": "ereymentau"})
    assert "error" in out and "SCADA" in out["error"]
    out = run_tool("no_such_tool", {})
    assert "error" in out


def test_api_health():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        r = c.get("/api/farms")
        assert {f["id"] for f in r.json()} == {"kelmarsh", "ereymentau"}
        r = c.get("/api/exports/..%2F..%2Fsecret.txt")
        assert r.status_code == 404
