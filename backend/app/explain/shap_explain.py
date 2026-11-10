"""SHAP арқылы болжамды түсіндіру: (11) f(x) = φ0 + Σ φ_j.

LightGBM медиана (P50) моделі түсіндіріледі (ансамбльдегі негізгі мүше). SHAP мәндері
нормаланған қуаттан МВт-қа аударылады. Белгілер мағыналық топтарға біріктіріледі.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from app.core.farms import get_farm

GROUPS = {
    # ws_hub_prev/next — көрші сағаттардағы сол жылдамдық (сағаттық орташа қуатқа сәйкестендіру),
    # сондықтан олар «жел жылдамдығы» тобына жатады
    "wind_speed": ["ws10", "ws80", "ws100", "ws120", "ws_hub", "ws_hub_rho", "p_phys", "gust10", "gust_factor",
                   "ws_hub_prev", "ws_hub_next"],
    "direction": ["u100", "v100", "wd_sin", "wd_cos"],
    "density": ["rho", "t2m", "sp"],
    "shear": ["alpha"],
    "trajectory": ["ws_hub_d1", "ws_hub_std3"],
    "calendar": ["hour_sin", "hour_cos", "doy_sin", "doy_cos"],
}
GROUP_NAMES = {
    "wind_speed": {"kk": "жел жылдамдығы (ступица биіктігі / 100 м)", "ru": "скорость ветра (высота ступицы / 100 м)"},
    "direction": {"kk": "жел бағыты", "ru": "направление ветра"},
    "density": {"kk": "ауа тығыздығы (температура, қысым)", "ru": "плотность воздуха (температура, давление)"},
    "shear": {"kk": "желдің биіктік бойынша ығысуы", "ru": "сдвиг ветра по высоте"},
    "trajectory": {"kk": "жел жылдамдығының өзгеру қарқыны", "ru": "динамика скорости ветра"},
    "calendar": {"kk": "тәулік уақыты мен маусым", "ru": "время суток и сезон"},
}
FEATURE_NAMES = {
    "ws100": {"kk": "100 м биіктіктегі жел жылдамдығы", "ru": "скорость ветра на 100 м"},
    "ws_hub": {"kk": "ступица биіктігіндегі жел жылдамдығы", "ru": "скорость ветра на высоте ступицы"},
    "ws_hub_rho": {"kk": "тығыздыққа түзетілген жел жылдамдығы", "ru": "скорость ветра с поправкой на плотность"},
    "p_phys": {"kk": "қуат қисығы бойынша физикалық болжам", "ru": "физический прогноз по кривой мощности"},
    "ws80": {"kk": "80 м биіктіктегі жел жылдамдығы", "ru": "скорость ветра на 80 м"},
    "ws120": {"kk": "120 м биіктіктегі жел жылдамдығы", "ru": "скорость ветра на 120 м"},
    "ws10": {"kk": "10 м биіктіктегі жел жылдамдығы", "ru": "скорость ветра на 10 м"},
    "gust10": {"kk": "жел екпіні", "ru": "порывы ветра"},
    "gust_factor": {"kk": "екпін коэффициенті", "ru": "коэффициент порывистости"},
    "alpha": {"kk": "желдің ығысу көрсеткіші", "ru": "показатель сдвига ветра"},
    "rho": {"kk": "ауа тығыздығы", "ru": "плотность воздуха"},
    "t2m": {"kk": "ауа температурасы", "ru": "температура воздуха"},
    "sp": {"kk": "атмосфералық қысым", "ru": "атмосферное давление"},
    "wd_sin": {"kk": "жел бағыты (sin)", "ru": "направление ветра (sin)"},
    "wd_cos": {"kk": "жел бағыты (cos)", "ru": "направление ветра (cos)"},
    "u100": {"kk": "желдің u-компоненті", "ru": "u-компонента ветра"},
    "v100": {"kk": "желдің v-компоненті", "ru": "v-компонента ветра"},
    "hour_sin": {"kk": "тәулік сағаты (sin)", "ru": "час суток (sin)"},
    "hour_cos": {"kk": "тәулік сағаты (cos)", "ru": "час суток (cos)"},
    "doy_sin": {"kk": "жыл маусымы (sin)", "ru": "сезон (sin)"},
    "doy_cos": {"kk": "жыл маусымы (cos)", "ru": "сезон (cos)"},
    "ws_hub_d1": {"kk": "жылдамдықтың сағаттық өзгерісі", "ru": "часовое изменение скорости"},
    "ws_hub_std3": {"kk": "жылдамдықтың тербелісі (3 сағ)", "ru": "изменчивость скорости (3 ч)"},
    "ws_hub_prev": {"kk": "алдыңғы сағаттағы жылдамдық", "ru": "скорость в предыдущий час"},
    "ws_hub_next": {"kk": "келесі сағаттағы жылдамдық", "ru": "скорость в следующий час"},
}


def _feature_group(f: str) -> str:
    return next((g for g, fs in GROUPS.items() if f in fs), "other")


def _explainer(bundle):
    lgbm = bundle.nwp_models["lgbm"]
    return lgbm, shap.TreeExplainer(lgbm.median_model)


def shap_values(bundle, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    lgbm, ex = _explainer(bundle)
    sv = ex.shap_values(X[lgbm.features])
    base = float(np.ravel(ex.expected_value)[0])
    return pd.DataFrame(sv, index=X.index, columns=lgbm.features), base


def global_importance(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Сурет 18: орташа |SHAP|, МВт."""
    cap = get_farm(bundle.farm_id).capacity_mw
    sv, _ = shap_values(bundle, X)
    imp = sv.abs().mean().mul(cap).sort_values(ascending=False)
    return pd.DataFrame({
        "feature": imp.index,
        "mean_abs_shap_mw": imp.to_numpy(),
        "name_kk": [FEATURE_NAMES.get(f, {}).get("kk", f) for f in imp.index],
        "group": [_feature_group(f) for f in imp.index],
    })


def explain_window(bundle, X: pd.DataFrame, reference: pd.DataFrame | None = None, lang: str = "kk", top: int = 4) -> dict:
    """Кезеңдегі болжамды түсіндіру.

    reference берілсе — кезеңнің referenceке қарағанда өзгеруіне (мысалы, кешкі төмендеу
    күндізгі деңгейге қарағанда) әр фактордың қосқан үлесі; әйтпесе — базалық деңгейге (φ0) қарағанда.
    """
    cap = get_farm(bundle.farm_id).capacity_mw
    sv, base = shap_values(bundle, X)
    pred = (base + sv.sum(axis=1)).clip(0, 1)
    if reference is not None and len(reference):
        sv_ref, _ = shap_values(bundle, reference)
        delta = sv.mean() - sv_ref.mean()
        ref_level = float((base + sv_ref.sum(axis=1)).clip(0, 1).mean() * cap)
        mode = "vs_reference"
    else:
        delta = sv.mean()
        ref_level = base * cap
        mode = "vs_baseline"
    by_group = delta.groupby([_feature_group(f) for f in delta.index]).sum().mul(cap)
    by_group = by_group.reindex(by_group.abs().sort_values(ascending=False).index)
    feats = delta.mul(cap).reindex(delta.abs().sort_values(ascending=False).index)
    factors = []
    for f, v in feats.head(top).items():
        factors.append({
            "feature": f,
            "name": FEATURE_NAMES.get(f, {}).get(lang, f),
            "contribution_mw": round(float(v), 2),
            "direction": "up" if v > 0 else "down",
            "mean_value": round(float(X[f].mean()), 2),
            **({"reference_value": round(float(reference[f].mean()), 2)} if reference is not None and f in reference else {}),
        })
    return {
        "mode": mode,
        "window_mean_mw": round(float(pred.mean() * cap), 2),
        "reference_mean_mw": round(ref_level, 2),
        "factors": factors,
        "groups": [
            {"group": g, "name": GROUP_NAMES.get(g, {}).get(lang, g), "contribution_mw": round(float(v), 2)}
            for g, v in by_group.head(top).items()
        ],
        "model": "LightGBM P50, SHAP TreeExplainer",
    }
