"""Станция бойынша оқытылған модельдер жинағы (bundle) және оның тізілімі.

Жұмыс режимінде (live) станцияның нақты уақыттағы SCADA деректері болмауы мүмкін, сондықтан
платформа NWP режимінің модельдерін қолданады: physical, lgbm, ensemble. Тарихи қуатты қажет
ететін модельдер (persistence, lgbm_short, LSTM/TFT, Chronos) эксперименттерде (бэктест) бағаланады.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from app.core.config import get_settings
from app.features.physics import PowerCurve
from app.models.base import Forecaster
from app.models.ensemble import combine

LIVE_MODELS = ("ensemble", "lgbm", "physical")


@dataclass
class FarmBundle:
    farm_id: str
    curve: PowerCurve
    nwp_models: dict[str, Forecaster]
    short_models: dict[str, Forecaster] = field(default_factory=dict)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def available_models(self) -> list[str]:
        names = list(self.nwp_models)
        if self.ensemble_weights:
            names = ["ensemble", *names]
        return names

    @property
    def best_model(self) -> str:
        return self.meta.get("best_model") or ("ensemble" if self.ensemble_weights else "lgbm")

    def predict_nwp(self, feats: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
        model = model or self.best_model
        if model == "ensemble":
            preds = {n: self.nwp_models[n].predict(feats) for n in self.ensemble_weights if n in self.nwp_models}
            return combine(preds, self.ensemble_weights)
        if model not in self.nwp_models:
            raise ValueError(f"Модель жоқ: {model}. Қолжетімді: {', '.join(self.available_models)}")
        return self.nwp_models[model].predict(feats)

    # ------------------------------------------------------------------ persistence
    def save(self, root: Path | None = None) -> Path:
        root = root or get_settings().models_dir
        d = root / self.farm_id
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "bundle.joblib")
        self.curve.to_json(d / "power_curve.json")
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return d


_CACHE: dict[str, FarmBundle] = {}


def bundle_path(farm_id: str) -> Path:
    return get_settings().models_dir / farm_id / "bundle.joblib"


def load_bundle(farm_id: str, reload: bool = False) -> FarmBundle:
    if farm_id in _CACHE and not reload:
        return _CACHE[farm_id]
    p = bundle_path(farm_id)
    if not p.exists():
        raise FileNotFoundError(
            f"'{farm_id}' станциясының модельдері табылмады. Алдымен scripts/03_train.py іске қосыңыз."
        )
    _CACHE[farm_id] = joblib.load(p)
    return _CACHE[farm_id]
