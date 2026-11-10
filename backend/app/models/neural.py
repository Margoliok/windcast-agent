"""M3 LSTM, M4 TFT (NeuralForecast) және M5 Chronos-Bolt (zero-shot) — зерттеу модельдері.

Көп горизонтты (h = 1…72) болжам: issue_time t0 сәтінде соңғы `input_size` сағаттық нақты
қуат + келешектегі NWP белгілері («болашақта белгілі кіріс», TFT идеясы) беріледі.
Бэктест: тест кезеңінде күн сайын 00:00 UTC-де болжам шығарылады; әр қадам үшін NWP
дереккөзі алдын ала болжау уақытына сәйкес таңдалады (select_by_lead).

Бұл модельдерге болжам сәтіндегі нақты қуат қажет, сондықтан олар жұмыс режимінде
(SCADA ағыны жоқ станцияларда) емес, эксперименттерде бағаланады.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.evaluation.metrics import QUANTILES
from app.features.build import select_by_lead
from app.ingest import dataset as ds

H = 72
EXOG = ["ws_hub_rho", "p_phys", "ws100", "ws10", "wd_sin", "wd_cos", "rho", "alpha", "gust10"]
QCOLS = [f"q{t:g}" for t in QUANTILES]


def _issue_times(test_period, step_h: int = 24) -> pd.DatetimeIndex:
    a, b = pd.Timestamp(test_period[0]), pd.Timestamp(test_period[1])
    return pd.date_range(a.normalize() + pd.Timedelta(days=1), b - pd.Timedelta(hours=H), freq=f"{step_h}h")


def _series_frame(y: pd.Series, feats0: pd.DataFrame) -> pd.DataFrame:
    """NeuralForecast форматы: unique_id, ds, y + экзогендік белгілер. y-тегі бос мәндер
    p_phys-пен толтырылады (тек оқыту контексті үшін; бағалау тек нақты y бойынша)."""
    idx = feats0.index
    yy = y.reindex(idx)
    filled = yy.fillna(feats0["p_phys"]).ffill().bfill()
    df = feats0[EXOG].ffill().bfill().copy()
    df.insert(0, "y", filled.to_numpy())
    df.insert(0, "ds", idx)
    df.insert(0, "unique_id", "kelmarsh")
    return df.reset_index(drop=True)


def _futr(sources, t0: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for h in range(1, H + 1):
        t = t0 + pd.Timedelta(hours=h)
        src = select_by_lead(sources, h)
        r = src[EXOG].reindex([t]).iloc[0] if t in src.index else pd.Series(np.nan, index=EXOG)
        rows.append({"unique_id": "kelmarsh", "ds": t, **r.to_dict()})
    f = pd.DataFrame(rows)
    f[EXOG] = f[EXOG].ffill().bfill()
    return f


def _long(preds: dict[pd.Timestamp, np.ndarray], y: pd.Series) -> pd.DataFrame:
    """{t0: (H × 9 кванттиль)} -> ұзын кесте valid_time, h, q…, y."""
    parts = []
    for t0, q in preds.items():
        vt = pd.date_range(t0 + pd.Timedelta(hours=1), periods=H, freq="1h")
        df = pd.DataFrame(np.clip(np.sort(q, axis=1), 0, 1), columns=QCOLS)
        df.insert(0, "valid_time", vt)
        df.insert(1, "h", np.arange(1, H + 1))
        df["y"] = y.reindex(vt).to_numpy()
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _col_quantile(col: str, prefix: str) -> float | None:
    """NeuralForecast бағандары: X-median, X-lo-80, X-hi-80, X-ql0.1 -> кванттиль деңгейі."""
    rest = col[len(prefix) + 1:]
    if rest == "median":
        return 0.5
    if rest.startswith("lo-"):
        return round((100 - float(rest[3:])) / 200, 4)
    if rest.startswith("hi-"):
        return round((100 + float(rest[3:])) / 200, 4)
    if rest.startswith("ql"):
        return float(rest[2:])
    return None


def _quantile_matrix(fc: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = {}
    for c in fc.columns:
        if c.startswith(prefix + "-"):
            q = _col_quantile(c, prefix)
            if q is not None:
                cols[q] = c
    return np.column_stack([fc[cols[min(cols, key=lambda k: abs(k - t))]].to_numpy() for t in QUANTILES])


def run_neuralforecast(bundle, sources, y, meta, max_steps: int = 600, log=print) -> dict[str, pd.DataFrame]:
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import LSTM, TFT

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    sp = meta["splits"]
    full = _series_frame(y, sources[0])
    train_end = pd.Timestamp(sp["val"][1])
    train_df = full[full.ds <= train_end]
    level_loss = MQLoss(quantiles=list(QUANTILES))
    models = [
        LSTM(h=H, input_size=72, futr_exog_list=EXOG, loss=level_loss, max_steps=max_steps,
             encoder_hidden_size=64, decoder_hidden_size=64, scaler_type="standard", random_seed=1,
             enable_progress_bar=False, enable_model_summary=False, logger=False),
        # CPU-да оқыту уақытын шектеу үшін: кіші терезелер батчы және жасырын қабат
        TFT(h=H, input_size=96, futr_exog_list=EXOG, loss=level_loss, max_steps=max_steps,
            hidden_size=32, windows_batch_size=128, batch_size=16, scaler_type="standard", random_seed=1,
            enable_progress_bar=False, enable_model_summary=False, logger=False),
    ]
    nf = NeuralForecast(models=models, freq="h")
    log(f"  NeuralForecast оқыту: {len(train_df)} сағат, max_steps={max_steps}")
    nf.fit(df=train_df)
    out = {"lstm": {}, "tft": {}}
    issues = _issue_times(sp["test"])
    log(f"  NeuralForecast бэктест: {len(issues)} болжам шығару сәті")
    for k, t0 in enumerate(issues):
        if k % 50 == 0:
            log(f"    {k}/{len(issues)}")
        hist = full[full.ds <= t0]
        fc = nf.predict(df=hist, futr_df=_futr(sources, t0))
        for name, cls in (("lstm", "LSTM"), ("tft", "TFT")):
            out[name][t0] = _quantile_matrix(fc, cls)
    return {k: _long(v, y) for k, v in out.items()}


def run_chronos(y: pd.Series, meta, allow_download: bool, context: int = 512, log=print) -> pd.DataFrame | None:
    import torch
    from chronos import BaseChronosPipeline

    name = os.environ.get("CHRONOS_MODEL", "amazon/chronos-bolt-small")
    from huggingface_hub import snapshot_download

    try:  # салмақтар жергілікті кэште бар ма? (жоқ болса және рұқсат берілмесе — жүктемейміз)
        path = snapshot_download(name, local_files_only=not allow_download)
    except Exception as e:
        log(f"  Chronos салмақтары жоқ ({name}); 01_download_data.py --full арқылы жүктеледі. ({type(e).__name__})")
        return None
    pipe = BaseChronosPipeline.from_pretrained(path, device_map="cpu", torch_dtype=torch.float32)
    yy = y.asfreq("1h").interpolate(limit=6).ffill().bfill()
    preds = {}
    for t0 in _issue_times(meta["splits"]["test"]):
        ctx = torch.tensor(yy.loc[:t0].tail(context).to_numpy(), dtype=torch.float32)
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=list(QUANTILES))
        preds[t0] = q[0].numpy()
    return _long(preds, y)


def run_neural_backtests(bundle, horizons, allow_download: bool = False, max_steps: int = 600, log=print) -> dict[str, pd.DataFrame]:
    """E5: LSTM, TFT, Chronos -> бэктест кестелері (тек берілген горизонттар)."""
    from app.evaluation.backtest import nwp_sources

    meta = ds.load_meta()
    y = ds.load_power()["power_norm"].asfreq("1h")
    sources = nwp_sources(bundle)
    res: dict[str, pd.DataFrame] = {}
    try:
        res.update(run_neuralforecast(bundle, sources, y, meta, max_steps=max_steps, log=log))
    except ImportError as e:
        log(f"  NeuralForecast жоқ: {e}")
    try:
        ch = run_chronos(y, meta, allow_download, log=log)
        if ch is not None:
            res["chronos"] = ch
    except ImportError as e:
        log(f"  Chronos кітапханасы жоқ: {e}")
    all_h = pd.concat([d.assign(model=k) for k, d in res.items()], ignore_index=True) if res else pd.DataFrame()
    if len(all_h):
        all_h.to_parquet(get_settings().processed_dir / "kelmarsh_neural_bt.parquet")
    return {k: d[d.h.isin(horizons)] for k, d in res.items()}
