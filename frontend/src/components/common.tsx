import createPlotlyComponent from "react-plotly.js/factory";
// @ts-ignore — plotly.js-dist-min типтері жоқ
import Plotly from "plotly.js-dist-min";
import { ReactNode, useEffect, useState } from "react";
import { Lang, t } from "../i18n";

export const Plot = createPlotlyComponent(Plotly);

export const C = {
  navy: "#0b3d5c",
  blue: "#2a7fb8",
  band: "rgba(127,184,212,0.35)",
  red: "#c0392b",
  orange: "#e39b3b",
  grey: "#9e9e9e",
  green: "#3f9e6e",
};

export const baseLayout = (extra: Record<string, any> = {}): any => ({
  autosize: true,
  margin: { l: 56, r: 56, t: 16, b: 48 },
  font: { family: "Inter, Segoe UI, sans-serif", size: 12, color: "#1d2b36" },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  legend: { orientation: "h", y: -0.18 },
  hovermode: "x unified",
  xaxis: { gridcolor: "#e8eef2" },
  yaxis: { gridcolor: "#e8eef2", rangemode: "tozero" },
  ...extra,
});

export function Card({ title, children, right, className = "" }: { title?: ReactNode; children: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header className="card-h">
          <h3>{title}</h3>
          <div>{right}</div>
        </header>
      )}
      {children}
    </section>
  );
}

export function Kpi({ label, value, unit, sub }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode }) {
  return (
    <div className="kpi">
      <div className="kpi-l">{label}</div>
      <div className="kpi-v">
        {value}
        {unit && <span className="kpi-u"> {unit}</span>}
      </div>
      {sub && <div className="kpi-s">{sub}</div>}
    </div>
  );
}

export function GateBadge({ gate, lang }: { gate?: { passed: boolean; checks?: any[] }; lang: Lang }) {
  if (!gate) return null;
  const title = (gate.checks || []).map((c: any) => `${c.passed ? "✓" : "✗"} ${c.check}: ${c.detail ?? ""}`).join("\n");
  return (
    <span className={`badge ${gate.passed ? "ok" : "bad"}`} title={title}>
      {t("gate", lang)}: {gate.passed ? t("gatePassed", lang) : t("gateFailed", lang)}
    </span>
  );
}

export function useApi<T = any>(fn: () => Promise<T>, deps: any[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e.message || e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading };
}

export function State({ loading, error, lang }: { loading: boolean; error: string | null; lang: Lang }) {
  if (loading) return <div className="muted pad">{t("loading", lang)}</div>;
  if (error) return <div className="err pad">{t("error", lang)}: {error}</div>;
  return null;
}

export function ForecastChart({ series, capacity, lang, height = 360, actual }: { series: any[]; capacity: number; lang: Lang; height?: number; actual?: any[] }) {
  const x = series.map((r) => r.time_local);
  const traces: any[] = [
    { x, y: series.map((r) => r.p90), type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x, y: series.map((r) => r.p10), type: "scatter", mode: "lines", fill: "tonexty", fillcolor: C.band, line: { width: 0 }, name: "P10–P90" },
    { x, y: series.map((r) => r.p50), type: "scatter", mode: "lines", line: { color: C.navy, width: 2.5 }, name: "P50" },
  ];
  if (series[0]?.ws_hub !== undefined)
    traces.push({ x, y: series.map((r) => r.ws_hub), type: "scatter", mode: "lines", yaxis: "y2", line: { color: C.red, width: 1, dash: "dot" }, name: t("wind", lang) });
  if (actual) traces.push({ x: actual.map((r) => r.time_local), y: actual.map((r) => r.actual), type: "scatter", mode: "lines", line: { color: C.red, width: 1.6 }, name: t("actual", lang) });
  return (
    <Plot
      data={traces}
      layout={baseLayout({
        height,
        yaxis: { title: t("power", lang), range: [0, capacity * 1.05], gridcolor: "#e8eef2" },
        yaxis2: { title: t("wind", lang), overlaying: "y", side: "right", showgrid: false, rangemode: "tozero" },
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: capacity, y1: capacity, line: { dash: "dash", color: "#aaa", width: 1 } }],
      })}
      useResizeHandler
      style={{ width: "100%" }}
      config={{ displaylogo: false, responsive: true }}
    />
  );
}

export function Table({ rows, cols }: { rows: any[]; cols: { key: string; label: string; fmt?: (v: any, r: any) => ReactNode }[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{cols.map((c) => <th key={c.key}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{cols.map((c) => <td key={c.key}>{c.fmt ? c.fmt(r[c.key], r) : r[c.key] ?? "—"}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export const fmt = (v: any, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
