"""Эксперименттер E1–E5 (3.2–3.3 бөлімдер). Нәтижелер reports/tables және data/processed ішіне жазылады.

E1 — горизонттар бойынша дәлдік (Кесте 14, Сурет 16, H1)
E2 — кіріс метеодеректер түрі: ERA5 vs NWP (Кесте 15)
E3 — ықтималдық болжамның калибрленуі (Кесте 16, H2)
E4 — SHAP маңыздылығы (Сурет 18)
E5 — нейрондық және foundation модельдер (LSTM, TFT, Chronos) — кітапханалар бар болса
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.farms import KELMARSH
from app.evaluation import metrics as M
from app.evaluation.backtest import KEY_HORIZONS, QCOLS, horizon_curve, metrics_table, run_backtest
from app.features.build import NWP_FEATURES
from app.ingest import dataset as ds
from app.models.lgbm import LGBMQuantileModel
from app.models.registry import load_bundle
from app.models.training import _mask, load_features_for

MODEL_NAMES = {
    "persistence": "M0 Persistence",
    "physical": "M1 Физикалық",
    "lgbm": "M2 LightGBM-Q",
    "lgbm_short": "M2 LightGBM-Q (1–6 сағ, лагтармен)",
    "lstm": "M3 LSTM",
    "tft": "M4 TFT",
    "chronos": "M5 Chronos-Bolt",
    "ensemble": "M6 Ансамбль",
}


def tables_dir():
    d = get_settings().reports_dir / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(df: pd.DataFrame, name: str, title: str) -> None:
    df.to_csv(tables_dir() / f"{name}.csv", index=False, encoding="utf-8-sig")
    try:
        md = df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        md = df.to_string(index=False)
    (tables_dir() / f"{name}.md").write_text(f"**{title}**\n\n{md}\n", encoding="utf-8")


def table13_splits() -> pd.DataFrame:
    meta = ds.load_meta()
    power = ds.load_power()
    rows = []
    purpose = {"train": "модельдерді оқыту, Optuna CV", "val": "ансамбль салмақтары, модель таңдау", "test": "соңғы бағалау"}
    names = {"train": "Оқыту", "val": "Валидация", "test": "Тест"}
    for k, (a, b) in meta["splits"].items():
        sub = power.loc[a:b]
        rows.append({
            "жиынтық": names[k],
            "кезең": f"{a[:10]} – {b[:10]}",
            "сағаттар": len(sub),
            "жарамды сағаттар": int(sub.power_norm.notna().sum()),
            "мақсаты": purpose[k],
        })
    t = pd.DataFrame(rows)
    _save(t, "tab13_splits", "Кесте 13 — Деректерді оқыту, валидация және тест кезеңдеріне бөлу")
    return t


RESEARCH_MODELS = ("lstm", "tft", "chronos")


def _tab14(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["model_name"] = t.model.map(MODEL_NAMES).fillna(t.model)
    return t[["model_name", "h", "mae", "rmse", "nmae", "nrmse", "bias", "skill", "n"]].rename(
        columns={"model_name": "модель", "h": "горизонт, сағ", "mae": "MAE, МВт", "rmse": "RMSE, МВт",
                 "nmae": "nMAE, %", "nrmse": "nRMSE, %", "bias": "bias, МВт", "skill": "SS"})


def e1_tables(bt: pd.DataFrame) -> pd.DataFrame:
    """Кесте 14 — жұмыс модельдері толық тест жиынтығында (сағат сайынғы болжам);
    Кесте 14b — зерттеу модельдерімен (LSTM, TFT, Chronos) бірге, күнделікті 00:00 UTC
    болжамдарының ортақ жиынтығында (әділ салыстыру үшін)."""
    main = metrics_table(bt[~bt.model.isin(RESEARCH_MODELS)])
    _save(_tab14(main), "tab14_horizons", "Кесте 14 — Модельдердің болжау горизонттары бойынша дәлдігі (толық тест жиынтығы)")
    main.to_csv(tables_dir() / "e1_metrics_full.csv", index=False)
    if bt.model.isin(RESEARCH_MODELS).any():
        allm = metrics_table(bt)
        _save(_tab14(allm), "tab14b_horizons_with_neural",
              "Кесте 14b — Зерттеу модельдерін (LSTM, TFT, Chronos) қоса салыстыру (күнделікті 00:00 UTC болжамдары, ортақ жиынтық)")
        allm.to_csv(tables_dir() / "e1_metrics_with_neural.csv", index=False)
        e3_calibration(allm, name="tab16b_calibration_with_neural",
                       title="Кесте 16b — Калибрлеу, зерттеу модельдерімен бірге (24 сағ, ортақ жиынтық)")
    return main


def e1_horizons(bundle, extra_models=None, log=print) -> pd.DataFrame:
    log("E1: горизонттар бойынша бэктест")
    bt = run_backtest(bundle, KEY_HORIZONS, extra_models=extra_models)
    bt.to_parquet(get_settings().processed_dir / "kelmarsh_backtest.parquet")
    full = e1_tables(bt)
    log("E1: nMAE(h) қисығы, h=1…72")
    curve = horizon_curve(bundle)
    curve.to_csv(tables_dir() / "fig16_nmae_by_horizon.csv", index=False)
    return full


def e2_weather_source(bundle, log=print) -> pd.DataFrame:
    """ERA5 vs NWP: бірдей LightGBM баптаулары, әр түрлі кіріс."""
    log("E2: ERA5 vs NWP")
    meta = ds.load_meta()
    sp = meta["splits"]
    y = ds.load_power()["power_norm"]
    params = bundle.meta.get("lgbm_params", {})
    F = {s: load_features_for(s, bundle.curve) for s in ("nwp", "era5")}

    def split(src):
        D = F[src].copy()
        D["y"] = y.reindex(D.index)
        D = D[D.y.notna() & D.ws100.notna()]
        trva = _mask(D.index, sp["train"]) | _mask(D.index, sp["val"])
        te = _mask(D.index, sp["test"])
        return D[trva], D[te]

    (nwp_tr, nwp_te), (era_tr, era_te) = split("nwp"), split("era5")
    common = nwp_te.index.intersection(era_te.index)
    nwp_te, era_te = nwp_te.loc[common], era_te.loc[common]
    m_nwp = LGBMQuantileModel(NWP_FEATURES, params).fit(nwp_tr, nwp_tr.y)
    m_era = LGBMQuantileModel(NWP_FEATURES, params).fit(era_tr, era_tr.y)
    cases = [
        ("NWP", "NWP", m_nwp, nwp_te, "жұмыс режиміне сәйкес (адал бағалау)"),
        ("ERA5", "ERA5", m_era, era_te, "«мінсіз болжам» — жоғарғы шек"),
        ("ERA5", "NWP", m_era, nwp_te, "реанализбен оқытып, болжам беру"),
    ]
    rows = []
    cap = KELMARSH.capacity_mw
    for tr_src, te_src, mdl, te, note in cases:
        q = mdl.predict(te) * cap
        yt = te.y.to_numpy() * cap
        rows.append({
            "оқыту кірісі": tr_src, "тест кірісі": te_src,
            "nMAE, %": M.nmae(yt, q["q0.5"], cap), "nRMSE, %": M.nrmse(yt, q["q0.5"], cap),
            "CRPS, %": 100 * M.crps_from_quantiles(yt / cap, q[QCOLS].to_numpy() / cap, M.QUANTILES),
            "PICP80, %": M.picp(yt, q["q0.1"], q["q0.9"]),
            "n": len(te), "түсініктеме": note,
        })
    t = pd.DataFrame(rows)
    _save(t, "tab15_era5_vs_nwp", "Кесте 15 — Кіріс метеодеректер түрінің болжам дәлдігіне әсері (ERA5 және NWP)")
    return t


def e3_calibration(full: pd.DataFrame, h: int = 24, name: str = "tab16_calibration",
                   title: str | None = None) -> pd.DataFrame:
    t = full[full.h == h].copy()
    t = t[t.model != "lgbm_short"] if h > 6 else t
    out = pd.DataFrame({
        "модель": t.model.map(MODEL_NAMES).fillna(t.model),
        "PICP (P10–P90), %": t.picp_80,
        "аралық ені, % P_inst": t.width_80,
        "CRPS, % P_inst": 100 * t.crps,
        "пинбол (P50), % P_inst": 100 * t.pinball_p50,
    })
    _save(out, name, title or f"Кесте 16 — Ықтималдық болжамның калибрлеу көрсеткіштері (горизонт {h} сағ)")
    return out


def reliability(bt: pd.DataFrame, h: int = 24) -> pd.DataFrame:
    """Әр кванттильдің эмпирикалық жиілігі (идеалда = τ)."""
    g = bt[(bt.h == h)].dropna(subset=["y", "q0.5"])
    rows = []
    for model, d in g.groupby("model"):
        for tau, c in zip(M.QUANTILES, QCOLS):
            rows.append({"model": model, "tau": tau, "observed": float((d.y <= d[c]).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(tables_dir() / "reliability_h24.csv", index=False)
    return t


def e4_shap(bundle, log=print, n: int = 3000) -> pd.DataFrame:
    log("E4: SHAP")
    from app.explain.shap_explain import global_importance

    meta = ds.load_meta()
    F = load_features_for("nwp", bundle.curve)
    te = F[_mask(F.index, meta["splits"]["test"])].dropna(subset=["ws100"])
    imp = global_importance(bundle, te.sample(min(n, len(te)), random_state=0))
    imp.to_csv(tables_dir() / "fig18_shap_importance.csv", index=False)
    return imp


def run_all(with_neural: bool = True, log=print) -> dict:
    bundle = load_bundle("kelmarsh", reload=True)
    table13_splits()
    extra = {}
    if with_neural:
        try:
            from app.models.neural import run_neural_backtests

            extra = run_neural_backtests(bundle, KEY_HORIZONS, log=log)
        except Exception as e:  # кітапхана немесе салмақтар жоқ
            log(f"E5 өткізілді: {e}")
    full = e1_horizons(bundle, extra_models=extra, log=log)
    e2 = e2_weather_source(bundle, log=log)
    e3 = e3_calibration(full)
    bt = pd.read_parquet(get_settings().processed_dir / "kelmarsh_backtest.parquet")
    reliability(bt)
    imp = e4_shap(bundle, log=log)
    summary = {
        "mode": ds.load_meta()["mode"],
        "best_by_horizon": {
            int(h): g.sort_values("nmae").iloc[0][["model", "nmae"]].to_dict() for h, g in full.groupby("h")
        },
        "h1_check": _h1(full),
        "h2_check": {m: float(v) for m, v in full[full.h == 24].set_index("model").picp_80.items()},
        "e2": e2.to_dict(orient="records"),
        "top_shap": imp.head(5).to_dict(orient="records"),
    }
    (tables_dir() / "experiments_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return summary


def _h1(full: pd.DataFrame) -> dict:
    """H1: NWP модельдері persistence-тен h > 3 сағ горизонттарда nMAE бойынша жақсы ма."""
    out = {}
    for h, g in full.groupby("h"):
        g = g.set_index("model")
        if "persistence" in g.index and "lgbm" in g.index:
            out[int(h)] = {
                "persistence_nmae": float(g.loc["persistence", "nmae"]),
                "lgbm_nmae": float(g.loc["lgbm", "nmae"]),
                "lgbm_better": bool(g.loc["lgbm", "nmae"] < g.loc["persistence", "nmae"]),
            }
    return out
