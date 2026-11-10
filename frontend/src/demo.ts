/**
 * GitHub Pages демо-режимі: бэкендсіз, статикалық снимоктан (public/demo/*.json) оқиды.
 * Барлық сандар нақты сервистерден scripts/08_export_demo.py арқылы алынған.
 * Тек кезеңге тәуелді тарих қорытындысы браузерде есептеледі (сол формулалармен).
 */
export const DEMO = import.meta.env.VITE_DEMO === "1";
export const BASE = import.meta.env.BASE_URL || "/";
export const REPO_URL = "https://github.com/Margoliok/windcast-agent";

const cache = new Map<string, Promise<any>>();

function load(file: string): Promise<any> {
  if (!cache.has(file)) {
    cache.set(
      file,
      fetch(`${BASE}demo/${file}`).then((r) => {
        if (!r.ok) throw new Error(`Демо-деректер жоқ: ${file}`);
        return r.json();
      }),
    );
  }
  return cache.get(file)!;
}

/** Файл сілтемесі: демода 'demo/exports/x' -> BASE + жол. */
export const fileUrl = (u?: string | null) => (!u ? "" : u.startsWith("demo/") ? `${BASE}${u}` : u);

const r1 = (v: number, d = 1) => Math.round(v * 10 ** d) / 10 ** d;

function inRange(t: string, a?: string, b?: string) {
  return (!a || t >= a) && (!b || t < b);
}

async function history(q: URLSearchParams) {
  if (q.get("farm_id") !== "kelmarsh")
    throw new Error("Бұл станцияның SCADA деректері ашық емес. Тарихи талдау тек Kelmarsh үшін қолжетімді.");
  const h = await load("history/kelmarsh.json");
  const a = q.get("start_local") ?? undefined;
  const b = q.get("end_local") ?? undefined;
  const rows = h.rows.filter((r: any[]) => inRange(r[0], a, b));
  if (!rows.length) throw new Error(`Бұл кезеңге деректер жоқ. Қолжетімді кезең: ${h.available_range.start_local} – ${h.available_range.end_local}.`);
  const raw = rows.map((r: any[]) => (r[1] == null ? null : Math.max(0, r[1])));
  const clean = rows.map((r: any[]) => r[2]).filter((v: any) => v != null) as number[];
  const ws = rows.map((r: any[]) => r[3]).filter((v: any) => v != null) as number[];
  const rawVals = raw.filter((v: any) => v != null) as number[];
  const sum = (x: number[]) => x.reduce((s, v) => s + v, 0);
  return {
    farm: h.farm,
    available_range: h.available_range,
    unit: "MW",
    summary: {
      hours: rows.length,
      valid_hours: clean.length,
      energy_mwh: r1(sum(rawVals)),
      mean_mw: clean.length ? r1(sum(clean) / clean.length, 2) : null,
      max_mw: clean.length ? r1(Math.max(...clean), 2) : null,
      capacity_factor_pct: rawVals.length ? r1((sum(rawVals) / rawVals.length / h.capacity_mw) * 100) : null,
      mean_ws100_era5: ws.length ? r1(sum(ws) / ws.length, 2) : null,
      outage_hours: rows.filter((r: any[]) => r[4] === "outage").length,
    },
    series: rows.map((r: any[]) => ({ time_local: r[0], power_mw: r[1], clean_power_mw: r[2], ws100_era5: r[3], status: r[4] })),
  };
}

async function downtime(q: URLSearchParams) {
  if (q.get("farm_id") !== "kelmarsh") throw new Error("SCADA деректері тек Kelmarsh үшін ашық.");
  const d = await load("history/kelmarsh_downtime.json");
  const a = q.get("start_local") ?? undefined;
  const b = q.get("end_local") ?? undefined;
  const periods = d.periods.filter((p: any) => inRange(p.start_local, a, b));
  return {
    ...d,
    period_local: `${a ?? ""} – ${b ?? ""}`,
    n_periods: periods.length,
    total_hours: periods.reduce((s: number, p: any) => s + p.duration_h, 0),
    total_lost_energy_mwh: r1(periods.reduce((s: number, p: any) => s + p.lost_energy_mwh, 0)),
    periods,
  };
}

async function whatif(q: URLSearchParams) {
  const farm = q.get("farm_id");
  const h = q.get("hours") || "24";
  const d = Math.round(Number(q.get("delta_wind_pct") || 0) / 5) * 5;
  if (d === 0) {
    const base = await load(`whatif/${farm}_${h}_5.json`);
    return {
      ...base, delta_wind_pct: 0, scenario: base.baseline, delta_energy_mwh_p50: 0, delta_energy_pct: 0,
      series: base.series.map((s: any) => ({ ...s, p50_scenario: s.p50_base })),
    };
  }
  return load(`whatif/${farm}_${h}_${Math.max(-50, Math.min(50, d))}.json`);
}

export async function demoApi(path: string): Promise<any> {
  const [p, query = ""] = path.split("?");
  const q = new URLSearchParams(query);
  const farm = q.get("farm_id") || "kelmarsh";
  const lang = q.get("lang") || "kk";
  switch (p) {
    case "/api/farms":
      return load("farms.json");
    case "/api/health":
      return load("health.json");
    case "/api/forecast":
      return load(`forecast/${farm}_${q.get("hours") || 72}_${q.get("model") || "auto"}.json`);
    case "/api/explain":
      return load(`explain/${farm}_${lang}.json`);
    case "/api/weather":
      return load(`weather/${farm}.json`);
    case "/api/whatif":
      return whatif(q);
    case "/api/schedule":
      return load(`schedule/${farm}_${lang}.json`);
    case "/api/reports":
      return load(`report/${farm}_${lang}.json`);
    case "/api/history":
      return history(q);
    case "/api/downtime":
      return downtime(q);
    case "/api/models/compare":
      return load(`models/compare_${q.get("horizon") || 24}.json`);
    case "/api/models/backtest_series":
      return load(`models/bt_${q.get("model") || "ensemble"}_${q.get("horizon") || 24}.json`);
    case "/api/models/horizons":
      return load("models/horizons.json");
    case "/api/agent/recordings":
      return load("agent_recordings.json");
  }
  if (p.startsWith("/api/experiments/table/")) return load(`tables/${p.split("/").pop()}.json`);
  throw new Error(`Демо-режимде қолжетімсіз: ${p}`);
}
