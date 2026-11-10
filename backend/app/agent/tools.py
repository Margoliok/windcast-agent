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
