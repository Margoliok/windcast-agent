import { useState } from "react";
import { api, qs } from "../api";
import { DEMO, fileUrl } from "../demo";
import { baseLayout, C, Card, ForecastChart, fmt, GateBadge, Kpi, Plot, State, Table, useApi } from "../components/common";
import { Lang, t } from "../i18n";

export function Schedule({ farm, lang, capacity }: { farm: string; lang: Lang; capacity: number }) {
  const [date, setDate] = useState("");
  const [ver, setVer] = useState(0);
  const sc = useApi(() => api(`/api/schedule${qs({ farm_id: farm, date_local: date, lang })}`), [farm, ver, lang]);
  const d: any = sc.data;
  return (
    <div className="grid">
      <Card className="span-12" title={`${t("nav_schedule", lang)}${d ? ` · ${d.farm} · ${d.date_local}` : ""}`} right={
        <form className="row gap" onSubmit={(e) => { e.preventDefault(); setVer((v) => v + 1); }}>
          {!DEMO && <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />}
          {!DEMO && <button className="btn">{t("build", lang)}</button>}
          {d?.files && <><a className="btn" href={fileUrl(d.files.xlsx)}>XLSX</a><a className="btn ghost" href={fileUrl(d.files.csv)}>CSV</a></>}
          {d && <GateBadge gate={d.gate} lang={lang} />}
        </form>
      }>
        <State loading={sc.loading} error={sc.error} lang={lang} />
        {DEMO && <div className="muted small">{t("demoDate", lang)}</div>}
        {d && (
          <>
            <div className="kpis">
              <Kpi label={t("energy", lang)} value={fmt(d.summary.energy_mwh_p50, 1)} unit={t("mwh", lang)} sub={`${t("energyRange", lang)}: ${fmt(d.summary.energy_mwh_p10, 1)} – ${fmt(d.summary.energy_mwh_p90, 1)}`} />
              <Kpi label={t("meanPower", lang)} value={fmt(d.summary.mean_mw_p50)} unit="МВт" />
              <Kpi label={t("maxPower", lang)} value={fmt(d.summary.max_mw_p50)} unit="МВт" sub={d.summary.max_time_local} />
              <Kpi label={t("issued", lang)} value={<span className="small">{d.issue_time_local}</span>} sub={d.timezone} />
            </div>
            <ForecastChart series={d.schedule.map((r: any) => ({ ...r, time_local: r.hour_local }))} capacity={capacity} lang={lang} height={300} />
            <Table rows={d.schedule} cols={[
              { key: "hour_local", label: t("hour", lang) }, { key: "p10", label: "P10, МВт", fmt: (v) => fmt(v, 3) },
              { key: "p50", label: "P50, МВт", fmt: (v) => <b>{fmt(v, 3)}</b> }, { key: "p90", label: "P90, МВт", fmt: (v) => fmt(v, 3) },
              { key: "ws_hub", label: t("wind", lang) },
            ]} />
          </>
        )}
      </Card>
    </div>
  );
}

export function History({ farm, lang }: { farm: string; lang: Lang }) {
  const [range, setRange] = useState({ start: "2019-01-01", end: "2019-02-01" });
  const [q, setQ] = useState(range);
  const kel = farm === "kelmarsh";
  const h = useApi(() => (kel ? api(`/api/history${qs({ farm_id: farm, start_local: q.start, end_local: q.end })}`) : Promise.resolve(null)), [farm, q]);
  const dt = useApi(() => (kel ? api(`/api/downtime${qs({ farm_id: farm, start_local: q.start, end_local: q.end })}`) : Promise.resolve(null)), [farm, q]);
  if (!kel) return <div className="warn">{t("noScada", lang)}</div>;
  const d: any = h.data;
  const w: any = dt.data;
  return (
    <div className="grid">
      <Card className="span-12" title={t("historyTitle", lang)} right={
        <form className="row gap" onSubmit={(e) => { e.preventDefault(); setQ(range); }}>
          <input type="date" value={range.start} onChange={(e) => setRange({ ...range, start: e.target.value })} />
          <input type="date" value={range.end} onChange={(e) => setRange({ ...range, end: e.target.value })} />
          <button className="btn">{t("show", lang)}</button>
        </form>
      }>
        <State loading={h.loading} error={h.error} lang={lang} />
        {d && (
          <>
            <div className="kpis">
              <Kpi label={`Σ, ${t("mwh", lang)}`} value={fmt(d.summary.energy_mwh, 1)} />
              <Kpi label={t("meanPower", lang)} value={fmt(d.summary.mean_mw)} unit="МВт" sub={`${t("cf", lang)}: ${fmt(d.summary.capacity_factor_pct, 1)}%`} />
              <Kpi label={t("maxPower", lang)} value={fmt(d.summary.max_mw)} unit="МВт" />
              <Kpi label="ERA5 ws100" value={fmt(d.summary.mean_ws100_era5)} unit="м/с" sub={`${d.available_range.start_local} – ${d.available_range.end_local}`} />
            </div>
            <Plot
              data={[
                { x: d.series.map((r: any) => r.time_local), y: d.series.map((r: any) => r.power_mw), type: "scatter", mode: "lines", name: t("actual", lang), line: { color: C.navy, width: 1.2 } },
                { x: d.series.map((r: any) => r.time_local), y: d.series.map((r: any) => r.ws100_era5), type: "scatter", mode: "lines", name: "ERA5 ws100, м/с", yaxis: "y2", line: { color: C.red, width: 1, dash: "dot" } },
              ]}
              layout={baseLayout({ height: 320, yaxis: { title: t("power", lang), gridcolor: "#e8eef2", rangemode: "tozero" }, yaxis2: { title: "м/с", overlaying: "y", side: "right", showgrid: false, rangemode: "tozero" } })}
              useResizeHandler style={{ width: "100%" }} config={{ displaylogo: false }}
            />
          </>
        )}
      </Card>
      <Card className="span-12" title={`${t("downtimeTitle", lang)}${w ? ` · ${t("periods", lang)}: ${w.n_periods} · ${t("lost", lang)}: ${w.total_lost_energy_mwh}` : ""}`}>
        <State loading={dt.loading} error={dt.error} lang={lang} />
        {w && <div className="muted small">{w.criterion}</div>}
        {w && (w.periods.length ? (
          <Table rows={w.periods} cols={[
            { key: "start_local", label: t("start", lang) }, { key: "end_local", label: t("end", lang) },
            { key: "duration_h", label: t("duration", lang) }, { key: "mean_ws100_era5", label: "ERA5 ws100" },
            { key: "expected_energy_mwh", label: t("expected", lang) }, { key: "lost_energy_mwh", label: t("lost", lang), fmt: (v) => <b>{v}</b> },
          ]} />
        ) : <div className="muted pad">{t("none", lang)}</div>)}
      </Card>
    </div>
  );
}

export function WhatIf({ farm, lang }: { farm: string; lang: Lang }) {
  const [delta, setDelta] = useState(15);
  const [hours, setHours] = useState(24);
  const [q, setQ] = useState({ delta: 15, hours: 24 });
  const r = useApi(() => api(`/api/whatif${qs({ farm_id: farm, delta_wind_pct: q.delta, hours: q.hours })}`), [farm, q]);
  const d: any = r.data;
  return (
    <div className="grid">
      <Card className="span-12" title={t("whatifTitle", lang)} right={
        <form className="row gap" onSubmit={(e) => { e.preventDefault(); setQ({ delta, hours }); }}>
          <label>{t("delta", lang)}: <b>{delta > 0 ? "+" : ""}{delta}%</b></label>
          <input type="range" min={-50} max={50} step={5} value={delta} onChange={(e) => setDelta(+e.target.value)} />
          <select value={hours} onChange={(e) => setHours(+e.target.value)}>
            {[24, 48, 72].map((h) => <option key={h} value={h}>{h} {t("hours", lang)}</option>)}
          </select>
          <button className="btn">{t("build", lang)}</button>
        </form>
      }>
        <State loading={r.loading} error={r.error} lang={lang} />
        {d && (
          <>
            <div className="kpis">
              <Kpi label={`${t("baseline", lang)} (P50)`} value={fmt(d.baseline.energy_mwh_p50, 1)} unit={t("mwh", lang)} />
              <Kpi label={`${t("scenario", lang)} (${d.delta_wind_pct > 0 ? "+" : ""}${d.delta_wind_pct}%)`} value={fmt(d.scenario.energy_mwh_p50, 1)} unit={t("mwh", lang)} />
              <Kpi label="Δ" value={`${d.delta_energy_mwh_p50 > 0 ? "+" : ""}${fmt(d.delta_energy_mwh_p50, 1)}`} unit={t("mwh", lang)} sub={`${d.delta_energy_pct > 0 ? "+" : ""}${fmt(d.delta_energy_pct, 1)}%`} />
              <GateBadge gate={d.gate} lang={lang} />
            </div>
            <Plot
              data={[
                { x: d.series.map((s: any) => s.time_local), y: d.series.map((s: any) => s.p50_base), type: "scatter", mode: "lines", name: t("baseline", lang), line: { color: C.navy, width: 2 } },
                { x: d.series.map((s: any) => s.time_local), y: d.series.map((s: any) => s.p50_scenario), type: "scatter", mode: "lines", name: t("scenario", lang), line: { color: C.orange, width: 2 } },
              ]}
              layout={baseLayout({ height: 320, yaxis: { title: t("power", lang), gridcolor: "#e8eef2", rangemode: "tozero" } })}
              useResizeHandler style={{ width: "100%" }} config={{ displaylogo: false }}
            />
            <div className="muted small">{t("whatifNote", lang)}</div>
          </>
        )}
      </Card>
    </div>
  );
}
