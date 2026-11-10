import { useState } from "react";
import { api, qs } from "../api";
import { baseLayout, C, Card, fmt, Plot, State, Table, useApi } from "../components/common";
import { Lang, MODEL_NAMES, t } from "../i18n";

const COLORS: Record<string, string> = {
  persistence: C.grey, physical: C.orange, lgbm: C.blue, lgbm_short: "#7cc3e8", ensemble: C.navy,
  lstm: "#8e5bb5", tft: "#c0457a", chronos: C.green,
};

export default function Models({ farm, lang }: { farm: string; lang: Lang }) {
  const [h, setH] = useState(24);
  const [model, setModel] = useState("ensemble");
  const kel = farm === "kelmarsh";
  const cmp = useApi(() => api(`/api/models/compare${qs({ farm_id: "kelmarsh", horizon: h })}`), [h]);
  const curve = useApi(() => api(`/api/models/horizons`), []);
  const bt = useApi(() => api(`/api/models/backtest_series${qs({ farm_id: "kelmarsh", model, horizon: h })}`), [model, h]);
  const era = useApi(() => api(`/api/experiments/table/tab15_era5_vs_nwp`), []);
  const cal = useApi(() => api(`/api/experiments/table/tab16_calibration`), []);

  const byModel: Record<string, any[]> = {};
  ((curve.data as any[]) ?? []).forEach((r) => (byModel[r.model] ??= []).push(r));

  return (
    <div className="grid">
      {!kel && <div className="span-12 warn">{t("noScada", lang)}</div>}
      <Card className="span-7" title={t("compareTitle", lang)} right={
        <select value={h} onChange={(e) => setH(+e.target.value)}>
          {[1, 6, 24, 48, 72].map((x) => <option key={x} value={x}>{x} {t("hours", lang)}</option>)}
        </select>
      }>
        <State loading={cmp.loading} error={cmp.error} lang={lang} />
        {cmp.data && (
          <>
            <Table rows={(cmp.data as any).table} cols={[
              { key: "model", label: t("model", lang), fmt: (v) => <b style={{ color: COLORS[v] }}>{MODEL_NAMES[v] ?? v}</b> },
              { key: "nmae_pct", label: "nMAE, %" }, { key: "nrmse_pct", label: "nRMSE, %" }, { key: "mae_mw", label: "MAE, МВт" },
              { key: "skill_vs_persistence", label: "Skill" }, { key: "crps_pct", label: "CRPS, %" }, { key: "picp_80_pct", label: "PICP80, %" },
            ]} />
            <div className="muted small">{(cmp.data as any).period_local} · n = {(cmp.data as any).n_hours}</div>
          </>
        )}
      </Card>
      <Card className="span-5" title={t("horizonCurve", lang)}>
        <State loading={curve.loading} error={curve.error} lang={lang} />
        <Plot
          data={Object.entries(byModel).map(([m, rows]) => ({
            x: rows.map((r) => r.h), y: rows.map((r) => r.nmae), type: "scatter", mode: "lines",
            name: MODEL_NAMES[m] ?? m, line: { color: COLORS[m], width: m === "ensemble" ? 3 : 1.6 },
          }))}
          layout={baseLayout({ height: 340, margin: { l: 56, r: 20, t: 10, b: 110 }, legend: { orientation: "h", y: -0.4 }, xaxis: { title: t("horizon", lang) + ", " + t("hours", lang), gridcolor: "#e8eef2" }, yaxis: { title: "nMAE, %", gridcolor: "#e8eef2", rangemode: "tozero" } })}
          useResizeHandler style={{ width: "100%" }} config={{ displaylogo: false }}
        />
      </Card>
      <Card className="span-12" title={t("backtestTitle", lang)} right={
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {["ensemble", "lgbm", "physical", "persistence", "lgbm_short", "lstm", "tft", "chronos"].map((m) => <option key={m} value={m}>{MODEL_NAMES[m]}</option>)}
        </select>
      }>
        <State loading={bt.loading} error={bt.error} lang={lang} />
        {bt.data && (bt.data as any).series?.length > 0 && (() => {
          const s = (bt.data as any).series;
          const x = s.map((r: any) => r.time_local);
          return (
            <Plot
              data={[
                { x, y: s.map((r: any) => r.p90), type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
                { x, y: s.map((r: any) => r.p10), type: "scatter", mode: "lines", fill: "tonexty", fillcolor: C.band, line: { width: 0 }, name: "P10–P90" },
                { x, y: s.map((r: any) => r.p50), type: "scatter", mode: "lines", line: { color: C.navy, width: 2 }, name: "P50" },
                { x, y: s.map((r: any) => r.actual), type: "scatter", mode: "lines", line: { color: C.red, width: 1.4 }, name: t("actual", lang) },
              ]}
              layout={baseLayout({ height: 320, yaxis: { title: t("power", lang), gridcolor: "#e8eef2", rangemode: "tozero" } })}
              useResizeHandler style={{ width: "100%" }} config={{ displaylogo: false }}
            />
          );
        })()}
      </Card>
      <Card className="span-6" title={t("calibration", lang)}>
        {cal.data && <Table rows={cal.data as any[]} cols={Object.keys((cal.data as any[])[0] ?? {}).map((k) => ({ key: k, label: k, fmt: (v: any) => (typeof v === "number" ? fmt(v) : v) }))} />}
      </Card>
      <Card className="span-6" title={t("era5", lang)}>
        {era.data && <Table rows={era.data as any[]} cols={["оқыту кірісі", "тест кірісі", "nMAE, %", "nRMSE, %", "CRPS, %", "PICP80, %"].map((k) => ({ key: k, label: k, fmt: (v: any) => (typeof v === "number" ? fmt(v) : v) }))} />}
      </Card>
    </div>
  );
}
