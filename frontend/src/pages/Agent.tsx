import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, ChatResult, chatStream, TraceEvent } from "../api";
import { DEMO, fileUrl } from "../demo";
import { Card, ForecastChart, fmt, GateBadge, Table } from "../components/common";
import { Lang, MODEL_NAMES, SUGGESTIONS, t } from "../i18n";

type Msg = { role: "user" | "assistant"; content: string; result?: ChatResult };

const TOOL_ICON: Record<string, string> = {
  get_weather_forecast: "🌬", get_history: "🗄", predict_power: "⚡", explain_forecast: "🔎", backtest: "📏",
  compare_models: "📊", what_if: "🔀", detect_downtime: "⛔", build_dayahead_schedule: "🗓", generate_report: "📄",
};

function Artifact({ a, lang, capacity }: { a: any; lang: Lang; capacity: number }) {
  if (a.tool === "predict_power" && a.series)
    return <ForecastChart series={a.series} capacity={a.capacity_mw ?? capacity} lang={lang} height={260} />;
  if (a.tool === "build_dayahead_schedule" && a.schedule)
    return (
      <div>
        <ForecastChart series={a.schedule.map((r: any) => ({ ...r, time_local: r.hour_local }))} capacity={capacity} lang={lang} height={240} />
        {a.files && <div className="row gap"><a className="btn" href={fileUrl(a.files.xlsx)}>XLSX</a><a className="btn ghost" href={fileUrl(a.files.csv)}>CSV</a></div>}
      </div>
    );
  if (a.tool === "compare_models" && a.table)
    return (
      <Table rows={a.table} cols={[
        { key: "model", label: t("model", lang), fmt: (v) => MODEL_NAMES[v] ?? v },
        { key: "nmae_pct", label: "nMAE, %" }, { key: "skill_vs_persistence", label: "SS" },
        { key: "crps_pct", label: "CRPS, %" }, { key: "picp_80_pct", label: "PICP, %" },
      ]} />
    );
  if (a.tool === "detect_downtime" && a.periods?.length)
    return (
      <Table rows={a.periods} cols={[
        { key: "start_local", label: t("start", lang) }, { key: "end_local", label: t("end", lang) },
        { key: "duration_h", label: t("duration", lang) }, { key: "lost_energy_mwh", label: t("lost", lang) },
      ]} />
    );
  if (a.tool === "generate_report" && a.files)
    return <div className="row gap">{a.files.pdf && <a className="btn" href={fileUrl(a.files.pdf)}>PDF</a>}<a className="btn ghost" href={fileUrl(a.files.markdown)}>Markdown</a></div>;
  if (a.tool === "get_weather_forecast" && a.low_wind_windows)
    return a.low_wind_windows.length ? (
      <Table rows={a.low_wind_windows} cols={[
        { key: "start_local", label: t("start", lang) }, { key: "end_local", label: t("end", lang) },
        { key: "duration_h", label: t("duration", lang) }, { key: "mean_ws_hub", label: "м/с" },
      ]} />
    ) : null;
  return null;
}

function TraceView({ trace, lang }: { trace: TraceEvent[]; lang: Lang }) {
  return (
    <ol className="trace">
      {trace.map((e, i) => {
        if (e.step === "llm")
          return (
            <li key={i} className="tr-llm">
              <b>LLM</b> {e.tool_calls?.length ? `→ ${e.tool_calls.map((c: any) => c.name).join(", ")}` : "→ жауап жобасы / черновик"}
              <span className="muted"> · {fmt(e.t, 1)} с</span>
            </li>
          );
        if (e.step === "tool")
          return (
            <li key={i} className={e.error ? "tr-err" : "tr-tool"}>
              {TOOL_ICON[e.name] ?? "🔧"} <b>{e.name}</b>
              <code>{JSON.stringify(e.args)}</code>
              {e.error ? <div className="err small">{e.error}</div> : e.gate && <GateBadge gate={e.gate} lang={lang} />}
              <span className="muted"> · {fmt(e.t, 2)} с</span>
            </li>
          );
        if (e.step === "gate") {
          const f = (e.checks || []).find((c: any) => c.check === "faithfulness");
          return (
            <li key={i} className={e.passed ? "tr-ok" : "tr-err"}>
              🛡 <b>{t("gate", lang)}</b>: {e.passed ? t("gatePassed", lang) : t("gateFailed", lang)}
              {f && <span className="muted"> · faithfulness = {fmt(f.score, 2)} ({f.n_numbers})</span>}
            </li>
          );
        }
        if (e.step === "safe_fail") return <li key={i} className="tr-err">⛔ safe_fail</li>;
        return null;
      })}
    </ol>
  );
}

function DemoAgent({ lang, capacity }: { lang: Lang; capacity: number }) {
  const [recs, setRecs] = useState<any[] | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [live, setLive] = useState<TraceEvent[]>([]);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api("/api/agent/recordings").then(setRecs).catch(() => setRecs([]));
  }, []);
  const list = (recs ?? []).filter((r) => r.lang === lang);

  async function play(r: any) {
    if (busy) return;
    setBusy(true);
    setMsgs((m) => [...m, { role: "user", content: r.question }]);
    setLive([]);
    for (const e of r.trace ?? []) {
      await new Promise((ok) => setTimeout(ok, 450));
      setLive((l) => [...l, e]);
    }
    await new Promise((ok) => setTimeout(ok, 300));
    setMsgs((m) => [...m, { role: "assistant", content: r.answer, result: r }]);
    setBusy(false);
  }
  const last = [...msgs].reverse().find((m) => m.result);
  const trace = busy ? live : last?.result?.trace ?? [];

  return (
    <div className="grid">
      <Card className="span-8 chat-card" title={t("nav_agent", lang)}>
        <div className="note">{list.length ? t("demoChat", lang) : t("demoNoRec", lang)}</div>
        <div className="chat">
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">
                <ReactMarkdown>{m.content}</ReactMarkdown>
                {m.result && (
                  <>
                    {m.result.artifacts?.map((a, j) => <Artifact key={j} a={a} lang={lang} capacity={capacity} />)}
                    <div className="meta">
                      <GateBadge gate={m.result.gate} lang={lang} /> · {m.result.tools_called.length} tools · {fmt(m.result.latency_s, 1)} с · {m.result.tokens} tok
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          {busy && <div className="msg assistant"><div className="bubble muted">{t("thinking", lang)}</div></div>}
        </div>
        <div className="suggest">
          {list.map((r, i) => (
            <button key={i} className="chip" disabled={busy} onClick={() => play(r)}>{r.question}</button>
          ))}
        </div>
      </Card>
      <Card className="span-4" title={t("trajectory", lang)}>
        {trace.length ? <TraceView trace={trace} lang={lang} /> : <div className="muted pad">{t("trajectoryHint", lang)}</div>}
      </Card>
    </div>
  );
}

export default function Agent(props: { farm: string; lang: Lang; llmReady: boolean; capacity: number }) {
  return DEMO ? <DemoAgent lang={props.lang} capacity={props.capacity} /> : <LiveAgent {...props} />;
}

function LiveAgent({ farm, lang, llmReady, capacity }: { farm: string; lang: Lang; llmReady: boolean; capacity: number }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<TraceEvent[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, busy]);

  async function send(q: string) {
    if (!q.trim() || busy) return;
    const history = msgs.map((m) => ({ role: m.role, content: m.content }));
    setMsgs((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setBusy(true);
    setLive([]);
    setSelected(null);
    try {
      await chatStream(
        { question: q, lang, farm_id: farm, history },
        (e) => setLive((l) => [...l, e]),
        (r) => setMsgs((m) => [...m, { role: "assistant", content: r.answer, result: r }]),
      );
    } catch (e: any) {
      setMsgs((m) => [...m, { role: "assistant", content: `${t("error", lang)}: ${e.message}` }]);
    } finally {
      setBusy(false);
    }
  }

  const lastIdx = [...msgs.keys()].reverse().find((i) => msgs[i].result);
  const shown = selected ?? lastIdx;
  const trace = busy ? live : shown !== undefined && shown !== null ? msgs[shown].result?.trace ?? [] : [];

  return (
    <div className="grid">
      <Card className="span-8 chat-card" title={t("nav_agent", lang)}>
        {!llmReady && <div className="warn">{t("llmMissing", lang)}</div>}
        <div className="chat">
          {msgs.length === 0 && (
            <div className="suggest">
              <div className="muted">{t("suggestions", lang)}:</div>
              {SUGGESTIONS[lang].map((s) => (
                <button key={s} className="chip" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          )}
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.role}`} onClick={() => m.result && setSelected(i)}>
              <div className="bubble">
                <ReactMarkdown>{m.content}</ReactMarkdown>
                {m.result && (
                  <>
                    {m.result.artifacts?.map((a, j) => <Artifact key={j} a={a} lang={lang} capacity={capacity} />)}
                    <div className="meta">
                      <GateBadge gate={m.result.gate} lang={lang} /> · {m.result.tools_called.length} tools · {fmt(m.result.latency_s, 1)} с · {m.result.tokens} tok
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          {busy && <div className="msg assistant"><div className="bubble muted">{t("thinking", lang)}</div></div>}
          <div ref={endRef} />
        </div>
        <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(input); }}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder={t("chatPlaceholder", lang)} disabled={busy} />
          <button className="btn" disabled={busy || !input.trim()}>{t("send", lang)}</button>
        </form>
      </Card>
      <Card className="span-4" title={t("trajectory", lang)}>
        {trace.length ? <TraceView trace={trace} lang={lang} /> : <div className="muted pad">{t("trajectoryHint", lang)}</div>}
      </Card>
    </div>
  );
}
