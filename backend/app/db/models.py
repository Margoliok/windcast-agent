"""Дерекқор құрылымы (Сурет 11, ER-диаграмма): 7 кесте."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FarmRow(Base):
    __tablename__ = "farms"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    capacity_mw: Mapped[float] = mapped_column(Float)
    hub_height_m: Mapped[float] = mapped_column(Float)
    n_turbines: Mapped[int] = mapped_column(Integer)
    turbine_model: Mapped[str] = mapped_column(String(128))
    timezone: Mapped[str] = mapped_column(String(64))
    power_curve: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ScadaMeasurement(Base):
    """Станцияның сағаттық өндірісі (тазалаудан кейін)."""
    __tablename__ = "scada_measurements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    farm_id: Mapped[str] = mapped_column(ForeignKey("farms.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    active_power_mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_power_mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    availability: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok | outage | missing
    __table_args__ = (UniqueConstraint("farm_id", "timestamp"), Index("ix_scada_farm_ts", "farm_id", "timestamp"))


class WeatherForecast(Base):
    __tablename__ = "weather_forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    farm_id: Mapped[str] = mapped_column(ForeignKey("farms.id"))
    issue_time: Mapped[datetime] = mapped_column(DateTime)
    valid_time: Mapped[datetime] = mapped_column(DateTime)
    model: Mapped[str] = mapped_column(String(32), default="best_match")
    wind_speed_10m: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_100m: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_120m: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_hub: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[float | None] = mapped_column(Float, nullable=True)
    gusts: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (Index("ix_wf_farm_issue", "farm_id", "issue_time"),)


class WeatherHistory(Base):
    """ERA5 реанализі (тарихи «нақты» ауа райы)."""
    __tablename__ = "weather_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    farm_id: Mapped[str] = mapped_column(ForeignKey("farms.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    wind_speed_10m: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_100m: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (UniqueConstraint("farm_id", "timestamp"), Index("ix_wh_farm_ts", "farm_id", "timestamp"))


class ForecastRow(Base):
    __tablename__ = "forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    farm_id: Mapped[str] = mapped_column(ForeignKey("farms.id"))
    issue_time: Mapped[datetime] = mapped_column(DateTime)
    valid_time: Mapped[datetime] = mapped_column(DateTime)
    model_version: Mapped[str] = mapped_column(String(64))
    p10: Mapped[float] = mapped_column(Float)
    p50: Mapped[float] = mapped_column(Float)
    p90: Mapped[float] = mapped_column(Float)
    gate_passed: Mapped[bool] = mapped_column(default=True)
    __table_args__ = (Index("ix_fc_farm_issue", "farm_id", "issue_time"),)


class MetricRow(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    farm_id: Mapped[str] = mapped_column(ForeignKey("farms.id"))
    model_version: Mapped[str] = mapped_column(String(64))
    period: Mapped[str] = mapped_column(String(64))
    horizon: Mapped[int] = mapped_column(Integer)
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    nmae: Mapped[float] = mapped_column(Float)
    crps: Mapped[float | None] = mapped_column(Float, nullable=True)
    picp: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentLog(Base):
    __tablename__ = "agent_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    lang: Mapped[str] = mapped_column(String(4), default="kk")
    question: Mapped[str] = mapped_column(Text)
    tools_called: Mapped[list | None] = mapped_column(JSON, nullable=True)
    answer: Mapped[str] = mapped_column(Text, default="")
    gate_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    system: Mapped[str] = mapped_column(String(16), default="agent")  # agent | llm_only
