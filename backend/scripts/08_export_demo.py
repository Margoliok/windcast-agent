"""8-қадам: GitHub Pages демосы үшін статикалық деректер снимогі -> frontend/public/demo/.

Барлық JSON файлдар нақты API-мен бірдей сервис функцияларынан алынады (сол сандар).
Демо статикалық: болжамдар снимок жасалған сәттегі Open-Meteo болжамы бойынша.

Іске қосу:  python scripts/08_export_demo.py
"""
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from app.core.config import get_settings
from app.core.db import init_db
from app.core.farms import FARMS
from app.models.registry import load_bundle
from app.services import forecast_service as fs
from app.services import history_service as hs
from app.services.db_sync import sync_all
from app.services.report import generate_report
from app.services.schedule_export import build_dayahead_schedule

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "demo"
HOURS = (24, 48, 72, 120, 168)
DELTAS = range(-50, 55, 5)
WHATIF_HOURS = (24, 48, 72)
HORIZONS = (1, 6, 24, 48, 72)
BT_MODELS = ("ensemble", "lgbm", "physical", "persistence", "lgbm_short", "lstm", "tft", "chronos")


def dump(name: str, obj) -> None:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":")), encoding="utf-8")


def rewrite_files(obj):
    """'/api/exports/x' сілтемелерін демо бумасына көшіріп, салыстырмалы жолға ауыстыру."""
    if isinstance(obj, dict):
        return {k: rewrite_files(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_files(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("/api/exports/"):
        name = obj.rsplit("/", 1)[-1]
        src = get_settings().exports_dir / name
        if src.exists():
            (OUT / "exports").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, OUT / "exports" / name)
        return f"demo/exports/{name}"
    return obj


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    init_db()
    sync_all()
    s = get_settings()

    farms = []
    for f in FARMS.values():
        d = f.to_dict()
        b = load_bundle(f.id)
        d.update(models=b.available_models, best_model=b.best_model, data_mode=b.meta.get("mode"))
        farms.append(d)
    dump("farms.json", farms)
    snapshot = fs.fmt_local(fs.now_utc(), FARMS["kelmarsh"])
    dump("health.json", {"status": "demo", "llm_configured": False, "llm_model": s.openai_model, "demo": True,
                         "snapshot_utc": fs.now_utc().isoformat(), "snapshot_local_kelmarsh": snapshot})

    for f in FARMS.values():
        print(f"== {f.id}")
        b = load_bundle(f.id)
        for h in HOURS:
            for m in ["", *b.available_models]:
                dump(f"forecast/{f.id}_{h}_{m or 'auto'}.json", fs.predict_power(f.id, h, m or None))
        for lang in ("kk", "ru"):
            dump(f"explain/{f.id}_{lang}.json", fs.explain_forecast(f.id, lang=lang))
            dump(f"schedule/{f.id}_{lang}.json", rewrite_files(build_dayahead_schedule(f.id, lang=lang)))
            dump(f"report/{f.id}_{lang}.json", rewrite_files(generate_report(f.id, 72, None, lang)))
        dump(f"weather/{f.id}.json", fs.weather_forecast(f.id, 168, 4.0, 6))
        for h in WHATIF_HOURS:
            for d in DELTAS:
                if d == 0:
                    continue
                dump(f"whatif/{f.id}_{h}_{d}.json", fs.what_if(f.id, d, h))

    # тарих (Kelmarsh): толық сағаттық қатар; жиынтықтарды браузер есептейді
    rng = hs.data_range("kelmarsh")
    farm = FARMS["kelmarsh"]
    full = hs.get_history("kelmarsh", rng["start_local"], (pd.Timestamp(rng["end_local"]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"))
    dump("history/kelmarsh.json", {"farm": full["farm"], "available_range": rng, "capacity_mw": farm.capacity_mw,
                                   "columns": ["time_local", "power_mw", "clean_power_mw", "ws100_era5", "status"],
                                   "rows": [[r["time_local"], r["power_mw"], r["clean_power_mw"], r["ws100_era5"], r["status"]]
                                            for r in full["series"]]})
    dt = hs.detect_downtime("kelmarsh", rng["start_local"], (pd.Timestamp(rng["end_local"]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"))
    dt["periods"] = dt["periods"]  # барлық кезең (браузер кезең бойынша сүзеді)
    dump("history/kelmarsh_downtime.json", dt)

    # модельдер
    for h in HORIZONS:
        dump(f"models/compare_{h}.json", hs.compare_models("kelmarsh", horizon=h))
        for m in BT_MODELS:
            try:
                dump(f"models/bt_{m}_{h}.json", hs.backtest_series("kelmarsh", m, h))
            except Exception as e:  # модель бұл горизонтта жоқ (мысалы, lgbm_short h>6)
                dump(f"models/bt_{m}_{h}.json", {"series": [], "error": str(e)})
    t = s.reports_dir / "tables"
    dump("models/horizons.json", pd.read_csv(t / "fig16_nmae_by_horizon.csv").to_dict(orient="records"))
    for name in ("tab15_era5_vs_nwp", "tab16_calibration", "tab14_horizons", "tab14b_horizons_with_neural"):
        if (t / f"{name}.csv").exists():
            dump(f"tables/{name}.json", pd.read_csv(t / f"{name}.csv").to_dict(orient="records"))
    if (t / "experiments_summary.json").exists():
        shutil.copy2(t / "experiments_summary.json", OUT / "tables" / "experiments_summary.json")

    rec = s.reports_dir / "agent_eval" / "demo_recordings.json"
    dump("agent_recordings.json", rewrite_files(json.loads(rec.read_text(encoding="utf-8"))) if rec.exists() else [])
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Дайын: {OUT} ({size:.1f} МБ), снимок: {snapshot} (Europe/London)")


if __name__ == "__main__":
    main()
