import { useEffect, useState } from "react";
import { api } from "./api";
import { BASE, DEMO, REPO_URL } from "./demo";
import Agent from "./pages/Agent";
import Dashboard from "./pages/Dashboard";
import Models from "./pages/Models";
import { History, Schedule, WhatIf } from "./pages/Tools";
import { Lang, t } from "./i18n";

type Page = "dashboard" | "agent" | "models" | "schedule" | "history" | "whatif";
const PAGES: Page[] = ["dashboard", "agent", "models", "schedule", "history", "whatif"];

export default function App() {
  const [lang, setLang] = useState<Lang>((localStorage.getItem("lang") as Lang) || "kk");
  const [page, setPage] = useState<Page>((location.hash.slice(1) as Page) || "dashboard");
  const [farms, setFarms] = useState<any[]>([]);
  const [farm, setFarm] = useState<string>(localStorage.getItem("farm") || "kelmarsh");
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    api("/api/farms").then(setFarms).catch(() => setFarms([]));
    api("/api/health").then(setHealth).catch(() => setHealth(null));
  }, []);
  useEffect(() => {
    localStorage.setItem("lang", lang);
    document.documentElement.lang = lang;
  }, [lang]);
  useEffect(() => {
    localStorage.setItem("farm", farm);
  }, [farm]);
  useEffect(() => {
    if (location.hash.slice(1) !== page) location.hash = page;
  }, [page]);
  useEffect(() => {
    const onHash = () => {
      const p = location.hash.slice(1) as Page;
      if (PAGES.includes(p)) setPage(p);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const info = farms.find((f) => f.id === farm);
  const cap = info?.capacity_mw ?? 12.3;

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <img src={`${BASE}favicon.svg`} alt="" />
          <div>
            <div className="brand-name">WindCast Agent</div>
            <div className="brand-sub">{t("appSubtitle", lang)}</div>
          </div>
        </div>
        <nav>
          {PAGES.map((p) => (
            <button key={p} className={page === p ? "active" : ""} onClick={() => setPage(p)}>
              {t(`nav_${p}` as any, lang)}
            </button>
          ))}
        </nav>
        <div className="row gap">
          <label className="farm-sel">
            {t("farm", lang)}{" "}
            <select value={farm} onChange={(e) => setFarm(e.target.value)}>
              {farms.map((f) => (
                <option key={f.id} value={f.id}>{f.name} · {f.capacity_mw} МВт</option>
              ))}
            </select>
          </label>
          <div className="lang">
            {(["kk", "ru"] as Lang[]).map((l) => (
              <button key={l} className={lang === l ? "active" : ""} onClick={() => setLang(l)}>{l === "kk" ? "ҚАЗ" : "РУС"}</button>
            ))}
          </div>
        </div>
      </header>
      {DEMO && (
        <div className="demo-banner">
          {t("demoBanner", lang)} <b>{health?.snapshot_local_kelmarsh ?? "—"}</b> (Europe/London) ·{" "}
          <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub</a>
        </div>
      )}
      <main>
        {page === "dashboard" && <Dashboard farm={farm} lang={lang} farmInfo={info} />}
        {page === "agent" && <Agent farm={farm} lang={lang} llmReady={!!health?.llm_configured} capacity={cap} />}
        {page === "models" && <Models farm={farm} lang={lang} />}
        {page === "schedule" && <Schedule farm={farm} lang={lang} capacity={cap} />}
        {page === "history" && <History farm={farm} lang={lang} />}
        {page === "whatif" && <WhatIf farm={farm} lang={lang} />}
      </main>
      <footer>
        WindCast Agent · Open-Meteo (CC-BY 4.0) · Kelmarsh wind farm data (Cubico, CC-BY-4.0) · LLM: {health?.llm_model ?? "—"}
      </footer>
    </div>
  );
}
