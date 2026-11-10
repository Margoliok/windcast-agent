"""Автоматты болжау циклі (Сурет 12) — APScheduler, әр сағат сайын.

1. Open-Meteo-дан станция координаттары бойынша 72 сағаттық болжамды жүктеу -> weather_forecasts
2. Белгілерді есептеу (ступица жылдамдығы, ρ, бағыт, уақыт)
3. Ең үздік модельмен P10/P50/P90 болжам
4. Тексеру шлюзі (физикалық шектеулер, кванттильдер реті)
5. forecasts кестесіне жазу
6. Дашборд жаңа болжамды /api/forecast/latest арқылы көрсетеді
7. Нақты SCADA келгенде қателік есептеліп metrics-ке жазылады (Kelmarsh-та SCADA ағыны жоқ —
   бұл қадам тарихи деректерде бэктест ретінде орындалады).
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.agent.gate import check_series
from app.core.config import get_settings
from app.core.db import session
from app.core.farms import FARMS
from app.db.models import ForecastRow, WeatherForecast
from app.models.registry import bundle_path, load_bundle
from app.services import forecast_service as fs

log = logging.getLogger("windcast.scheduler")
_sched: BackgroundScheduler | None = None


def forecast_cycle() -> dict:
    done = {}
    for farm in FARMS.values():
        if not bundle_path(farm.id).exists():
            continue
        try:
            fc = fs.run_forecast(farm.id, 72)
            rows = fc.rows()
            gate = check_series(rows, farm.capacity_mw, expected_len=len(rows))
            version = f"{fc.model}@{load_bundle(farm.id).meta.get('trained_at', '?')}"
            with session() as s:
                for ts, r in fc.frame.iterrows():
                    s.add(WeatherForecast(farm_id=farm.id, issue_time=fc.issue_time.to_pydatetime(),
                                          valid_time=ts.to_pydatetime(), wind_speed_100m=float(r.ws100),
                                          wind_speed_hub=float(r.ws_hub),
                                          wind_direction=None if r.wd100 != r.wd100 else float(r.wd100),
                                          temperature=float(r.t2m)))
                    s.add(ForecastRow(farm_id=farm.id, issue_time=fc.issue_time.to_pydatetime(), valid_time=ts.to_pydatetime(),
                                      model_version=version, p10=float(r.p10), p50=float(r.p50), p90=float(r.p90),
                                      gate_passed=gate.passed))
            done[farm.id] = {"rows": len(rows), "gate": gate.passed}
        except Exception as e:  # бір станцияның қатесі циклді тоқтатпайды
            log.warning("forecast cycle failed for %s: %s", farm.id, e)
            done[farm.id] = {"error": str(e)}
    log.info("forecast cycle: %s", done)
    return done


def start() -> None:
    global _sched
    s = get_settings()
    if not s.scheduler_enabled or _sched is not None:
        return
    _sched = BackgroundScheduler(timezone="UTC")
    _sched.add_job(forecast_cycle, "interval", minutes=s.scheduler_interval_minutes, id="forecast",
                   max_instances=1, coalesce=True)
    _sched.add_job(forecast_cycle, "date", id="forecast_initial")  # іске қосылғанда бір рет
    _sched.start()


def stop() -> None:
    global _sched
    if _sched:
        _sched.shutdown(wait=False)
        _sched = None
