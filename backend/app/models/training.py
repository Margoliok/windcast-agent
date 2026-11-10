"""Модельдерді оқыту: қуат қисығы, M0–M2, ансамбль, Ерейментауға көшірілетін модель.

Схема:
  1. Қуат қисығы оқыту кезеңінде құрылады (SCADA режимінде — гондола анемометрі бойынша
     турбиналардың нақты қисығы; есептегіш режимінде — NWP ступица жылдамдығы бойынша).
  2. LightGBM гиперпараметрлері Optuna арқылы rolling-origin CV-мен (оқыту кезеңі ішінде,
     уақыт бойынша кеңейетін терезе) таңдалады — валидация мен тест жиынтықтары қолданылмайды.
  3. Ансамбль салмақтары валидация кезеңінде (train-де оқытылған модельдер) таңдалады.
  4. Соңғы модельдер train+val бойынша қайта оқытылады; тест кезеңі тек бағалау үшін.
"""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
import pandas as pd

from app.core.config import get_settings
from app.core.farms import EREYMENTAU, KELMARSH
from app.evaluation.metrics import pinball
from app.features.build import (
    NWP_FEATURES,
    SHORT_FEATURES,
    nwp_dataset,
    persistence_frame,
    short_dataset,
    weather_features,
)
from app.features.physics import PowerCurve
from app.ingest import dataset as ds
from app.models.ensemble import combine, fit_weights
from app.models.lgbm import DEFAULT_PARAMS, LGBMQuantileModel
from app.models.persistence import PersistenceModel
from app.models.physical import PhysicalModel
from app.models.registry import FarmBundle

# Станцияға тәуелсіз белгілер (абсолюттік қысым, температура, бағыт, жергілікті уақыт жоқ) —
# модельді басқа станцияға (Ерейментау) көшіру үшін.
TRANSFER_FEATURES = [
    "ws10", "ws80", "ws100", "ws120", "gust10", "gust_factor", "rho", "alpha",
    "ws_hub", "ws_hub_rho", "p_phys", "ws_hub_d1", "ws_hub_std3", "ws_hub_prev", "ws_hub_next",
]

_FLAT = PowerCurve(np.array([0.0, 40.0]), np.array([0.0, 0.0]), "placeholder")


def _mask(index, period) -> np.ndarray:
    a, b = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    return np.asarray((index >= a) & (index <= b))


def fit_curve(mode: str, power: pd.DataFrame, feats_raw: pd.DataFrame, train_period) -> PowerCurve:
    m = _mask(power.index, train_period)
    if mode == "scada" and "nacelle_ws" in power and power["nacelle_ws"].notna().sum() > 1000:
        return PowerCurve.fit(power.loc[m, "nacelle_ws"], power.loc[m, "power_norm"], source="scada_nacelle")
    v = feats_raw["ws_hub_rho"].reindex(power.index)[m]
    return PowerCurve.fit(v, power.loc[m, "power_norm"], source="nwp_hub_speed")


def make_features(wx: pd.DataFrame, hub: float, curve: PowerCurve) -> pd.DataFrame:
    return weather_features(wx, hub, curve)


def rolling_origin_folds(index: pd.DatetimeIndex, n_folds: int = 3, min_train_frac: float = 0.4):
    """Кеңейетін терезелі уақыттық CV (Сурет 7): [train | val] жылжиды."""
    n = len(index)
    start = int(n * min_train_frac)
    step = (n - start) // n_folds
    for k in range(n_folds):
        tr_end = start + k * step
        va_end = min(start + (k + 1) * step, n)
        yield np.arange(0, tr_end), np.arange(tr_end, va_end)


def _fit_persistence(y_all: pd.Series, sp: dict) -> PersistenceModel:
    """Persistence қателігінің кванттильдері 1..72 сағат горизонттары үшін (train+val)."""
    P = persistence_frame(y_all, range(1, 73))
    vt = pd.DatetimeIndex(P["valid_time"])
    P = P[(_mask(vt, sp["train"]) | _mask(vt, sp["val"])) & P.y.notna() & P.p_now.notna()]
    return PersistenceModel().fit(P.reset_index(drop=True), P.y.reset_index(drop=True))


def tune_lgbm(X: pd.DataFrame, y: pd.Series, features: list[str], n_trials: int = 25, log=print) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    m = y.notna().to_numpy()
    X, y = X[m], y[m]
    folds = list(rolling_origin_folds(X.index))

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        import lightgbm as lgb

        scores = []
        for tr, va in folds:
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=0,
                                    **{**DEFAULT_PARAMS, **params})
            mdl.fit(X.iloc[tr][features], y.iloc[tr])
            scores.append(pinball(y.iloc[va], mdl.predict(X.iloc[va][features]), 0.5))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    log(f"    Optuna: ең үздік pinball(0.5) = {study.best_value:.4f}, {study.best_params}")
    return study.best_params


def train_all(n_trials: int = 25, log=print) -> dict:
    t0 = time.time()
    meta = ds.load_meta()
    mode, sp = meta["mode"], meta["splits"]
    power = ds.load_power()
    wx = ds.load_weather("nwp")
    hub = KELMARSH.hub_height_m

    log("1) Қуат қисығы")
    raw = make_features(wx, hub, _FLAT)
    curve = fit_curve(mode, power, raw, sp["train"])
    feats = make_features(wx, hub, curve)
    y_all = power["power_norm"]

    D = nwp_dataset(y_all, feats).dropna(subset=["ws100"])
    tr, va = _mask(D.index, sp["train"]), _mask(D.index, sp["val"])
    trva = tr | va
    Dtr, Dva, Dtrva = D[tr & D.y.notna()], D[va & D.y.notna()], D[trva & D.y.notna()]
    log(f"   NWP жиынтығы: train={len(Dtr)}, val={len(Dva)}")

    log("2) LightGBM гиперпараметрлерін баптау (rolling-origin CV, train ішінде)")
    best = tune_lgbm(Dtr, Dtr.y, NWP_FEATURES, n_trials=n_trials, log=log)
    best_tr = tune_lgbm(Dtr, Dtr.y, TRANSFER_FEATURES, n_trials=max(10, n_trials // 2), log=log)

    log("3) Валидация: ансамбль салмақтары")
    val_models = {
        "physical": PhysicalModel().fit(Dtr, Dtr.y),
        "lgbm": LGBMQuantileModel(NWP_FEATURES, best).fit(Dtr, Dtr.y),
    }
    offsets = val_models["lgbm"].calibrate(Dva, Dva.y)
    log(f"   Конформдық түзетулер (P10, P90): {offsets.get(0.1, 0):+.4f}, {offsets.get(0.9, 0):+.4f}")
    val_preds = {n: m.predict(Dva) for n, m in val_models.items()}
    weights = fit_weights(val_preds, Dva.y)
    log(f"   Ансамбль салмақтары: {weights}")
    val_scores = {n: pinball(Dva.y, p["q0.5"], 0.5) for n, p in val_preds.items()}
    val_scores["ensemble"] = pinball(Dva.y, combine(val_preds, weights)["q0.5"], 0.5)

    tr_models = {
        "physical": val_models["physical"],
        "lgbm": LGBMQuantileModel(TRANSFER_FEATURES, best_tr).fit(Dtr, Dtr.y),
    }
    offsets_tr = tr_models["lgbm"].calibrate(Dva, Dva.y)
    weights_tr = fit_weights({n: m.predict(Dva) for n, m in tr_models.items()}, Dva.y)

    log("4) Соңғы модельдер (train+val)")
    nwp_models = {
        "physical": PhysicalModel().fit(Dtrva, Dtrva.y),
        "lgbm": LGBMQuantileModel(NWP_FEATURES, best).fit(Dtrva, Dtrva.y),
    }
    nwp_models["lgbm"].offsets = offsets
    S = short_dataset(y_all, feats).dropna(subset=["ws100", "p_now"])
    s_issue = pd.DatetimeIndex(S["valid_time"])
    Str = S[(_mask(s_issue, sp["train"]) | _mask(s_issue, sp["val"])) & S.y.notna()]
    short_models = {
        "persistence": _fit_persistence(y_all, sp),
        "lgbm_short": LGBMQuantileModel(SHORT_FEATURES, best).fit(Str, Str.y),
    }
    best_name = min(val_scores, key=val_scores.get)
    period_txt = {k: v for k, v in sp.items()}
    kel = FarmBundle(
        farm_id="kelmarsh",
        curve=curve,
        nwp_models=nwp_models,
        short_models=short_models,
        ensemble_weights=weights,
        meta={
            "mode": mode,
            "splits": period_txt,
            "curve_source": curve.source,
            "lgbm_params": best,
            "conformal_offsets": offsets,
            "val_pinball_p50": val_scores,
            "best_model": best_name,
            "features": NWP_FEATURES,
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
    )
    kel.save()

    log("5) Ерейментау: станцияға тәуелсіз белгілермен көшірілген модель")
    ere_lgbm = LGBMQuantileModel(TRANSFER_FEATURES, best_tr).fit(Dtrva, Dtrva.y)
    ere_lgbm.offsets = offsets_tr
    ere = FarmBundle(
        farm_id="ereymentau",
        curve=curve,
        nwp_models={
            "physical": nwp_models["physical"],
            "lgbm": ere_lgbm,
        },
        ensemble_weights=weights_tr,
        meta={
            "mode": mode,
            "transferred_from": "kelmarsh",
            "features": TRANSFER_FEATURES,
            "lgbm_params": best_tr,
            "best_model": "ensemble",
            "curve_source": curve.source,
            "note": EREYMENTAU.notes,
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
    )
    ere.save()
    _log_mlflow(kel.meta, val_scores)
    log(f"Дайын ({time.time() - t0:.0f} с). Ең үздік модель (валидация): {best_name}")
    return kel.meta


def _log_mlflow(meta: dict, val_scores: dict) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri((get_settings().data_dir / "mlruns").as_uri())
        mlflow.set_experiment("windcast-kelmarsh")
        with mlflow.start_run(run_name=f"train-{meta['mode']}"):
            mlflow.log_params({f"lgbm_{k}": v for k, v in meta["lgbm_params"].items()})
            mlflow.log_param("mode", meta["mode"])
            mlflow.log_param("curve_source", meta["curve_source"])
            for k, v in val_scores.items():
                mlflow.log_metric(f"val_pinball_p50_{k}", float(v))
            mlflow.log_dict(meta, "meta.json")
    except Exception as e:  # MLflow міндетті емес
        print(f"   (MLflow тіркелмеді: {e})")


def load_features_for(source: str, curve: PowerCurve, hub: float | None = None) -> pd.DataFrame:
    return make_features(ds.load_weather(source), hub or KELMARSH.hub_height_m, curve)


if __name__ == "__main__":
    print(json.dumps(train_all(), ensure_ascii=False, indent=2, default=str))
