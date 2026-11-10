import { useState } from "react";
import { api, qs } from "../api";
import { fileUrl } from "../demo";
import { baseLayout, C, Card, ForecastChart, fmt, GateBadge, Kpi, Plot, State, useApi } from "../components/common";
import { Lang, MODEL_NAMES, t } from "../i18n";

export default function Dashboard({ farm, lang, farmInfo }: { farm: string; lang: Lang; farmInfo: any }) {
  const [hours, setHours] = useState(72);
  const [model, setModel] = useState<string>("");
  const [reporting, setReporting] = useState(false);
  async function report() {
    setReporting(true);
    try {
      const r: any = await api(`/api/reports${qs({ farm_id: farm, hours: 72, lang })}`, { method: "POST" });
      window.open(fileUrl(r.files.pdf || r.files.markdown), "_blank");
    } catch (e: any) {
      alert(e.message);
    } finally {
      setReporting(false);
    }
  }
  const fc = useApi(() => api(`/api/forecast${qs({ farm_id: farm, hours, model })}`), [farm, hours, model]);
  const ex = useApi(() => api(`/api/explain${qs({ farm_id: farm, lang })}`), [farm, lang]);
  const wx = useApi(() => api(`/api/weather${qs({ farm_id: farm, hours: 168, low_wind_threshold: 4, min_duration_h: 6 })}`), [farm]);
  const d: any = fc.data;
  const s = d?.summary;
  const groups: any[] = (ex.data as any)?.groups ?? [];

  return (
    <div className="grid">
      <Card
        className="span-12"
        title={`${farmInfo?.name ?? farm} · ${t("forecastTitle", lang)}`}
        right={
          <div className="row gap">
            <label>
              {t("horizon", lang)}{" "}
              <select value={hours} onChange={(e) => setHours(+e.target.value)}>
                {[24, 48, 72, 120, 168].map((h) => (
                  <option key={h} value={h}>{h} {t("hours", lang)}</option>
                ))}
              </select>
            </label>
            <label>
              {t("model", lang)}{" "}
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                <option value="">{MODEL_NAMES[farmInfo?.best_model] ?? "auto"} (auto)</option>
                {(farmInfo?.models ?? []).map((m: string) => (
                  <option key={m} value={m}>{MODEL_NAMES[m] ?? m}</option>
                ))}
              </select>
            </label>
            {d && <GateBadge gate={d.gate} lang={lang} />}
            <button className="btn ghost" onClick={report} disabled={reporting}>📄 {t("report", lang)}</button>
          </div>
        }
      >
        <State loading={fc.loading} error={fc.error} lang={lang} />
        {s && (
          <>
            <div className="kpis">
              <Kpi label={`${t("energy", lang)}, ${hours} ${t("hours", lang)}`} value={fmt(s.energy_mwh_p50, 1)} unit={t("mwh", lang)}
                   sub={`${t("energyRange", lang)}: ${fmt(s.energy_mwh_p10, 1)} – ${fmt(s.energy_mwh_p90, 1)}`} />
              <Kpi label={t("meanPower", lang)} value={fmt(s.mean_mw_p50)} unit="МВт" sub={`${t("cf", lang)}: ${fmt(s.capacity_factor_pct, 1)}%`} />
              <Kpi label={t("maxPower", lang)} value={fmt(s.max_mw_p50)} unit="МВт" sub={s.max_time_local} />
              <Kpi label={t("meanWind", lang)} value={fmt(s.mean_ws_hub)} unit="м/с" sub={`${t("capacity", lang)}: ${d.capacity_mw} МВт`} />
            </div>
            <ForecastChart series={d.series} capacity={d.capacity_mw} lang={lang} />
            <div className="muted small">
              {t("issued", lang)}: {d.issue_time_local} ({d.timezone}) · {t("model", lang)}: {MODEL_NAMES[d.model] ?? d.model} ·{" "}
              {t("dataNote", lang)}: {d.model_info?.data_mode}
            </div>
          </>
        )}
      </Card>

      <Card className="span-7" title={t("explainTitle", lang)}>
        <State loading={ex.loading} error={ex.error} lang={lang} />
        {groups.length > 0 && (
          <Plot
            data={[{
              type: "bar", orientation: "h",
              y: groups.map((g) => g.name).reverse(),
              x: groups.map((g) => g.contribution_mw).reverse(),
              marker: { color: groups.map((g) => (g.contribution_mw >= 0 ? C.blue : C.orange)).reverse() },
            }]}
            layout={baseLayout({ height: 240, margin: { l: 260, r: 20, t: 10, b: 40 }, xaxis: { title: t("contribution", lang), zeroline: true }, yaxis: { automargin: true }, hovermode: "closest" })}
            useResizeHandler style={{ width: "100%" }} config={{ displaylogo: false }}
          />
        )}
        {ex.data && (
          <div className="muted small">
            {(ex.data as any).window_local} · P50 {fmt((ex.data as any).window_mean_mw)} МВт vs {fmt((ex.data as any).reference_mean_mw)} МВт (φ₀)
          </div>
        )}
      </Card>

      <Card className="span-5" title={t("lowWind", lang)}>
        <State loading={wx.loading} error={wx.error} lang={lang} />
        {wx.data && ((wx.data as any).low_wind_windows?.length ? (
          <ul className="list">
            {(wx.data as any).low_wind_windows.map((w: any, i: number) => (
              <li key={i}>
                <b>{w.start_local}</b> → {w.end_local} · {w.duration_h} {t("hours", lang)} · {w.mean_ws_hub} м/с
              </li>
            ))}
          </ul>
        ) : <div className="muted pad">{t("none", lang)} (7 тәулік / 7 суток)</div>)}
        {farmInfo?.notes?.[lang] && <p className="note">{farmInfo.notes[lang]}</p>}
      </Card>
    </div>
  );
}
