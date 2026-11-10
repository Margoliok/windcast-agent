import { DEMO, demoApi } from "./demo";

export type Row = Record<string, any>;

export async function api<T = any>(path: string, opts: RequestInit = {}): Promise<T> {
  if (DEMO) return demoApi(path);
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = j.detail ?? JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return r.json();
}

export const qs = (o: Record<string, any>) =>
  "?" +
  Object.entries(o)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
    .join("&");

export type TraceEvent = { node?: string; step: string; [k: string]: any };
export type ChatResult = {
  answer: string;
  status: string;
  gate?: { passed: boolean; checks: Row[] };
  trace: TraceEvent[];
  artifacts: Row[];
  tools_called: string[];
  latency_s: number;
  tokens: number;
  session_id: string;
};

/** SSE ағыны: әр қадамда onTrace, соңында onFinal. */
export async function chatStream(
  body: Row,
  onTrace: (e: TraceEvent) => void,
  onFinal: (r: ChatResult) => void,
): Promise<void> {
  const r = await fetch("/api/agent/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok || !r.body) throw new Error(r.statusText);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split(/\r?\n\r?\n/);
    buf = parts.pop() ?? "";
    for (const part of parts) {
      let ev = "message";
      const data: string[] = [];
      for (const line of part.split(/\r?\n/)) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      const payload = JSON.parse(data.join("\n"));
      if (ev === "trace") onTrace(payload);
      else if (ev === "final") onFinal(payload);
    }
  }
}
