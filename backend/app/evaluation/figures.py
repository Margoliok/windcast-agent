"""Дипломдық жұмыстың суреттері (кодпен салынатындары) -> reports/figures/*.png.

Сурет 5  — «жылдамдық–қуат» диаграммасы: тазалауға дейін және кейін
Сурет 6  — маусымдық және тәуліктік профильдер
Сурет 7  — rolling-origin кросс-валидация схемасы
Сурет 13 — ЖИ-агенттің LangGraph күйлер графы
Сурет 16 — nMAE-дің горизонтқа тәуелділігі
Сурет 17 — 72 сағаттық ықтималдық болжам мысалы (тест кезеңі)
Сурет 18 — SHAP маңыздылығы
Сурет 20 — Ерейментау ЖЭС келесі тәулікке болжамы (+ Кесте 18)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.farms import KELMARSH  # noqa: E402
from app.evaluation.experiments import MODEL_NAMES, tables_dir  # noqa: E402
from app.ingest import dataset as ds  # noqa: E402
from app.models.registry import load_bundle  # noqa: E402
from app.models.training import _mask, load_features_for  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"persistence": "#9e9e9e", "physical": "#e39b3b", "lgbm": "#2a7fb8", "lgbm_short": "#7cc3e8",
          "ensemble": "#0b3d5c", "lstm": "#8e5bb5", "tft": "#c0457a", "chronos": "#3f9e6e"}
BAND = "#8fc1dc"


def fig_dir():
    d = get_settings().reports_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name):
    p = fig_dir() / name
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig05_power_curve():
    meta = ds.load_meta()
    bundle = load_bundle("kelmarsh")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    if meta["mode"] == "scada" and (get_settings().processed_dir / "kelmarsh_scada_sample.parquet").exists():
        s = pd.read_parquet(get_settings().processed_dir / "kelmarsh_scada_sample.parquet")
        x_all, y_all = s.ws, s.power_kw / 2050
        x_cl, y_cl = s.ws[s.clean], s.power_kw[s.clean] / 2050
        xlabel = "Гондоладағы жел жылдамдығы, м/с"
        ylab = "P / P_ном (турбина)"
    else:
        p = ds.load_power()
        f = load_features_for("nwp", bundle.curve).reindex(p.index)
        x_all, y_all = f.ws_hub_rho, p.power_raw_mw / KELMARSH.capacity_mw
        x_cl, y_cl = f.ws_hub_rho, p.power_norm
        xlabel = "Ступица биіктігіндегі жел (NWP, ρ-түзетілген), м/с"
        ylab = "P / P_inst (станция)"
    removed = pd.Series(np.asarray(y_all), dtype=float).notna().to_numpy() & pd.Series(np.asarray(y_cl), dtype=float).isna().to_numpy()
    for ax, (x, y, t) in zip(axes, [(x_all, y_all, "а) тазалауға дейін"), (x_cl, y_cl, "б) тазалаудан кейін")]):
        ax.scatter(x, y, s=2, alpha=0.15, color="#2a7fb8", rasterized=True)
        if ax is axes[0] and removed.any():
            ax.scatter(np.asarray(x)[removed], np.asarray(y)[removed], s=5, alpha=0.6, color="#c0392b",
                       label=f"алынып тасталған ({removed.sum()} нүкте)", rasterized=True)
            ax.legend(loc="upper left", fontsize=8)
        g = np.linspace(0, 25, 200)
        ax.plot(g, bundle.curve(g), color="#c0392b", lw=2, label=f"эмпирикалық қисық ({bundle.curve.source})")
        ax.set_xlim(0, 25)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(t)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylab)
    axes[1].legend(loc="lower right", fontsize=8)
    return _save(fig, "fig05_power_curve_cleaning.png")


def fig06_profiles():
    p = ds.load_power()
    w = ds.load_weather("era5").reindex(p.index)
    loc = p.index.tz_localize("UTC").tz_convert(KELMARSH.timezone)
    df = pd.DataFrame({"p": p.power_mw.to_numpy(), "ws": w.ws100.to_numpy(), "m": loc.month, "h": loc.hour})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, xl, ticks in [(axes[0], "m", "Ай", range(1, 13)), (axes[1], "h", "Тәулік сағаты (жергілікті)", range(0, 24, 3))]:
        g = df.groupby(key).mean()
        ax.bar(g.index, g.p, color="#8fc1dc", label="орташа қуат, МВт")
        ax.set_xlabel(xl)
        ax.set_ylabel("МВт")
        ax.set_xticks(list(ticks))
        ax2 = ax.twinx()
        ax2.plot(g.index, g.ws, color="#c0392b", marker="o", ms=3, label="жел 100 м (ERA5), м/с")
        ax2.set_ylabel("м/с")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_title("а) маусымдық профиль")
    axes[1].set_title("б) тәуліктік профиль")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axes[1].get_figure().axes[-1].get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    return _save(fig, "fig06_seasonal_diurnal_profiles.png")


def fig07_rolling_origin(n_folds=3):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    total = 10
    for k in range(n_folds):
        tr = 4 + 2 * k
        y = n_folds - k
        ax.barh(y, tr, left=0, color="#2a7fb8", height=0.5)
        ax.barh(y, 2, left=tr, color="#e39b3b", height=0.5)
        ax.text(-0.2, y, f"{k + 1}-фолд", ha="right", va="center")
    ax.barh(0, 10, left=0, color="#d0d0d0", height=0.5)
    ax.barh(0, 2, left=10, color="#5a9e6f", height=0.5)
    ax.barh(0, 2, left=12, color="#c0392b", height=0.5)
    ax.text(-0.2, 0, "соңғы схема", ha="right", va="center")
    ax.set_xlim(-2.5, 14.5)
    ax.set_yticks([])
    ax.set_xlabel("уақыт →")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="#2a7fb8", label="CV: оқыту"), Patch(color="#e39b3b", label="CV: бағалау"),
                       Patch(color="#d0d0d0", label="train"), Patch(color="#5a9e6f", label="val"),
                       Patch(color="#c0392b", label="test")], ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    ax.spines["left"].set_visible(False)
    _ = total
    return _save(fig, "fig07_rolling_origin_cv.png")


def fig13_agent_graph():
    from app.agent.graph import export_graph_png

    export_graph_png(fig_dir() / "fig13_agent_graph.png")  # .mmd мәтіні
    nodes = {"START": (0.5, 5.2), "agent\n(LLM: жоспарлау,\nReAct)": (0.5, 3.9), "tools\n(10 құрал)": (2.6, 3.9),
             "verify\n(тексеру шлюзі)": (0.5, 2.3), "safe_fail\n(ашық қате)": (2.6, 1.0), "END": (0.5, 0.2)}
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    pos = {}
    for name, (x, y) in nodes.items():
        w, h = (1.5, 0.75) if "\n" in name else (0.9, 0.45)
        color = {"START": "#e0e0e0", "END": "#e0e0e0"}.get(name, "#d6eaf5")
        if name.startswith("verify"):
            color = "#fde2c8"
        if name.startswith("safe_fail"):
            color = "#f6d0d0"
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=color, ec="#333"))
        ax.text(x, y, name, ha="center", va="center", fontsize=9)
        pos[name.split("\n")[0]] = (x, y, w, h)

    def arrow(a, b, label="", rad=0.0, at=None):
        xa, ya, wa, ha = pos[a]
        xb, yb, wb, hb = pos[b]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=12,
                                     connectionstyle=f"arc3,rad={rad}", color="#333", shrinkA=22, shrinkB=22))
        if label:
            x, y = at or ((xa + xb) / 2, (ya + yb) / 2)
            ax.text(x, y, label, fontsize=8, ha="center", va="center", color="#333",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    arrow("START", "agent")
    arrow("agent", "tools", "tool_calls", rad=-0.3, at=(1.55, 4.45))
    arrow("tools", "agent", "нәтижелер", rad=-0.3, at=(1.55, 3.35))
    arrow("agent", "verify", "жауап жобасы", at=(0.05, 3.1))
    arrow("verify", "END", "өтті", at=(0.3, 1.25))
    arrow("verify", "agent", "өтпеді:\nқайталау ≤ 2", rad=0.6, at=(1.45, 3.05))
    arrow("verify", "safe_fail", "лимит", at=(1.75, 1.75))
    arrow("safe_fail", "END")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.3, 5.6)
    ax.axis("off")
    return _save(fig, "fig13_agent_graph.png")


def fig16_horizon():
    t = pd.read_csv(tables_dir() / "fig16_nmae_by_horizon.csv")
    nb = get_settings().processed_dir / "kelmarsh_neural_bt.parquet"
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for model, g in t.groupby("model"):
        ax.plot(g.h, g.nmae, label=MODEL_NAMES.get(model, model), color=COLORS.get(model), lw=2 if model == "ensemble" else 1.4)
    if nb.exists():
        from app.evaluation.backtest import metrics_table

        n = pd.read_parquet(nb)
        mt = metrics_table(n, common=False)
        for model, g in mt.groupby("model"):
            ax.plot(g.h, g.nmae, label=MODEL_NAMES.get(model, model), color=COLORS.get(model), lw=1.4, ls="--")
    ax.set_xlabel("Болжау горизонты, сағ")
    ax.set_ylabel("nMAE, % P_inst")
    ax.set_xticks([1, 6, 12, 24, 36, 48, 60, 72])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    if ds.load_meta()["mode"] == "grid_meter":
        ax.set_title("Ескерту: этап A — NWP архиві қысқа алдын ала болжам уақыты бойынша (горизонтқа тәуелсіз)", fontsize=8)
    return _save(fig, "fig16_nmae_by_horizon.png")


def fig17_probabilistic_example(model: str = "ensemble"):
    meta = ds.load_meta()
    bundle = load_bundle("kelmarsh")
    F = load_features_for("nwp", bundle.curve)
    y = ds.load_power().power_mw
    te = F[_mask(F.index, meta["splits"]["test"])].dropna(subset=["ws100"])
    # ең өзгермелі 72 сағаттық терезе (көрнекілік үшін), нақты деректер толық болуы керек
    yy = y.reindex(te.index)
    roll = yy.rolling(72).std()
    ok = yy.notna().rolling(72).sum() == 72
    end = roll[ok].idxmax()
    win = te.loc[end - pd.Timedelta(hours=71): end]
    q = bundle.predict_nwp(win, model) * KELMARSH.capacity_mw
    fig, ax = plt.subplots(figsize=(9, 3.8))
    loc = win.index.tz_localize("UTC").tz_convert(KELMARSH.timezone).tz_localize(None)
    ax.fill_between(loc, q["q0.1"], q["q0.9"], color=BAND, alpha=0.6, label="P10–P90")
    ax.plot(loc, q["q0.5"], color="#0b3d5c", lw=2, label=f"P50 ({MODEL_NAMES.get(model, model)})")
    ax.plot(loc, y.reindex(win.index), color="#c0392b", lw=1.5, label="нақты қуат")
    ax.axhline(KELMARSH.capacity_mw, color="#999", ls="--", lw=0.8)
    ax.set_ylabel("МВт")
    ax.set_ylim(0, KELMARSH.capacity_mw * 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    cov = float(((y.reindex(win.index) >= q["q0.1"]) & (y.reindex(win.index) <= q["q0.9"])).mean() * 100)
    ax.set_title(f"Kelmarsh, {loc[0]:%Y-%m-%d} – {loc[-1]:%Y-%m-%d}; терезедегі қамту {cov:.0f}%", fontsize=9)
    fig.autofmt_xdate()
    return _save(fig, "fig17_probabilistic_72h.png")


def fig18_shap():
    imp = pd.read_csv(tables_dir() / "fig18_shap_importance.csv")
    from app.explain.shap_explain import GROUP_NAMES, _feature_group

    imp["group"] = imp.feature.map(_feature_group)
    top = imp.head(12).iloc[::-1]
    grp = imp.groupby("group").mean_abs_shap_mw.sum().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].barh(top.name_kk, top.mean_abs_shap_mw, color="#2a7fb8")
    axes[0].set_xlabel("орташа |SHAP|, МВт")
    axes[0].set_title("а) жеке белгілер (12 үздік)")
    axes[1].barh([GROUP_NAMES.get(g, {}).get("kk", g) for g in grp.index], grp.values, color="#e39b3b")
    axes[1].set_xlabel("Σ орташа |SHAP|, МВт")
    axes[1].set_title("б) белгілер топтары")
    for ax in axes:
        ax.grid(alpha=0.3, axis="x")
    return _save(fig, "fig18_shap_importance.png")


def fig20_ereymentau():
    from app.services.schedule_export import build_dayahead_schedule

    s = build_dayahead_schedule("ereymentau", export=True)
    df = pd.DataFrame(s["schedule"])
    t = pd.to_datetime(df.hour_local)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(t, df.p10, df.p90, color=BAND, alpha=0.6, label="P10–P90")
    ax.plot(t, df.p50, color="#0b3d5c", lw=2, label="P50 (ансамбль)")
    ax.axhline(45, color="#999", ls="--", lw=0.8)
    ax.set_ylabel("МВт")
    ax.set_ylim(0, 47)
    ax2 = ax.twinx()
    ax2.plot(t, df.ws_hub, color="#c0392b", lw=1, ls=":", label="жел (ступица), м/с")
    ax2.set_ylim(0, max(12.0, float(df.ws_hub.max()) * 1.15))
    ax2.set_ylabel("м/с")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title(f"Ерейментау ЖЭС (45 МВт), {s['date_local']}, болжам шыққан уақыт {s['issue_time_local']} (Asia/Almaty)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    p = _save(fig, "fig20_ereymentau_dayahead.png")
    tab = df.rename(columns={"hour_local": "сағат", "p10": "P10, МВт", "p50": "P50, МВт", "p90": "P90, МВт", "ws_hub": "жел (ступица), м/с"})
    tab.to_csv(tables_dir() / "tab18_ereymentau_schedule.csv", index=False, encoding="utf-8-sig")
    (tables_dir() / "tab18_ereymentau_schedule.md").write_text(
        f"**Кесте 18 — Ерейментау ЖЭС тәуліктік өндіріс кестесі ({s['date_local']}, шыққан уақыты {s['issue_time_local']})**\n\n"
        + tab.to_string(index=False) + f"\n\nΣ P50 = {s['summary']['energy_mwh_p50']} МВт·сағ "
        f"(P10–P90: {s['summary']['energy_mwh_p10']} – {s['summary']['energy_mwh_p90']})\n", encoding="utf-8")
    (tables_dir() / "tab18_meta.json").write_text(json.dumps(s["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def reliability_plot():
    p = tables_dir() / "reliability_h24.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    for model, g in t.groupby("model"):
        if model == "lgbm_short":
            continue
        ax.plot(g.tau, g.observed, marker="o", ms=3, label=MODEL_NAMES.get(model, model), color=COLORS.get(model))
    ax.set_xlabel("номиналдық кванттиль τ")
    ax.set_ylabel("бақыланған жиілік P(y ≤ q_τ)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, "figA_reliability_h24.png")


def make_all(log=print, live: bool = True) -> list:
    out = []
    for fn in (fig05_power_curve, fig06_profiles, fig07_rolling_origin, fig13_agent_graph, fig16_horizon,
               fig17_probabilistic_example, fig18_shap, reliability_plot):
        try:
            out.append(fn())
            log(f"  ✓ {fn.__name__}")
        except Exception as e:
            log(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
    if live:
        try:
            out.append(fig20_ereymentau())
            log("  ✓ fig20_ereymentau (+ Кесте 18)")
        except Exception as e:
            log(f"  ✗ fig20_ereymentau: {e}")
    return out
