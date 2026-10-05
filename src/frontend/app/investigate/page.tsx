"use client";
import { useState } from "react";
import { Search, RotateCw, ServerOff } from "lucide-react";
import { analyze, anomaly, rootCause, ApiError } from "@/lib/api/client";
import type { AnalyzeFull } from "@/lib/types/api";
import { RiskCard, Shap, Candidates } from "@/components/investigation/Parts";
import { AnomalyPanel, WhatIfPanel } from "@/components/investigation/DemoPanels";
import Stepper, { Step } from "@/components/ui/Stepper";
import { InvestigationSkeleton } from "@/components/ui/Skeleton";

const DEMO_IDS = [100, 200, 300, 400, 500];
const STEPS: Step[] = [
  { label: "Detect", source: "live" }, { label: "Predict", source: "live" }, { label: "Explain", source: "live" },
  { label: "Root cause", source: "live" }, { label: "Simulate", source: "live" }, { label: "Recommend", source: "live" },
];

export default function Investigate() {
  const [id, setId] = useState("100");
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [data, setData] = useState<AnalyzeFull | null>(null);
  const [anomalyData, setAnomalyData] = useState<import("@/lib/types/api").AnomalyInfo | null>(null);
  const [rootCauseBusy, setRootCauseBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);

  async function run(raw = id) {
    const n = Number(raw);
    if (raw.trim() === "" || !Number.isInteger(n) || n < 0) { setErr(new ApiError("validation", "Enter a whole number of 0 or more.")); setState("error"); return; }
    setState("loading"); setErr(null);
    try { const [analysis, liveAnomaly] = await Promise.all([analyze(n, 5), anomaly(n)]); setData(analysis); setAnomalyData(liveAnomaly); setState("done"); }
    catch (e) { setErr(e instanceof ApiError ? e : new ApiError("server", "Unexpected error.")); setState("error"); }
  }
  const top = data?.root_cause_candidates?.[0];

  async function refreshRootCause() {
    if (data?.sample.sample_id == null) return;
    setRootCauseBusy(true);
    try {
      const rc = await rootCause(data.sample.sample_id, 5);
      setData((prev) => prev ? { ...prev, root_cause_candidates: rc.root_cause_candidates, anomaly: rc.anomaly, historical_matches: rc.historical_matches } : prev);
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError("server", "Unable to refresh root cause."));
    } finally { setRootCauseBusy(false); }
  }
  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">Investigate a sample</h1>
        <p className="mt-1 max-w-2xl text-sm text-zinc-400">Pick a SECOM row. The engine predicts failure risk, explains it, and ranks the sensors most associated with it.</p>
      </header>

      <form className="card mb-4 flex flex-wrap items-end gap-3 p-4" onSubmit={(e) => { e.preventDefault(); run(); }}>
        <label className="eyebrow">Sample ID (0-based row)
          <input value={id} onChange={(e) => setId(e.target.value)} inputMode="numeric" placeholder="e.g. 100" className="field w-44 num" />
        </label>
        <button className="btn-primary" disabled={state === "loading"}><Search size={15} />Why did this fail?</button>
        <div className="flex flex-wrap items-center gap-2 sm:ml-auto" role="group" aria-label="Demo samples">
          <span className="text-xs text-zinc-500">Try</span>
          {DEMO_IDS.map((d) => <button type="button" key={d} onClick={() => { setId(String(d)); run(String(d)); }} className="btn-ghost num">{d}</button>)}
        </div>
      </form>

      <Stepper steps={STEPS} done={state === "done" ? 6 : 0} running={state === "loading"} />

      <div className="mt-4" aria-live="polite">
        {state === "idle" && <div className="card grid place-items-center px-6 py-16 text-center"><p className="text-zinc-200">No sample loaded</p><p className="mt-1 text-sm text-zinc-500">Enter a sample ID or choose one of the demo samples above.</p></div>}
        {state === "loading" && <InvestigationSkeleton />}
        {state === "error" && err && (
          <div className="card flex flex-col items-start gap-3 border-risk/40 p-6" role="alert">
            <ServerOff className="text-risk" size={20} />
            <div><h2 className="font-semibold text-risk">{err.kind === "offline" ? "Root-cause engine offline" : "Unable to analyze sample"}</h2>
              <p className="mt-1 text-sm text-zinc-300">{err.message}</p></div>
            <button onClick={() => run()} className="btn-ghost"><RotateCw size={14} />Retry</button>
          </div>)}
        {state === "done" && data && (
          <div className="grid animate-rise gap-4 lg:grid-cols-3">
            <div className="card flex flex-wrap gap-x-6 gap-y-1 px-4 py-3 text-sm lg:col-span-3">
              <span className="num text-zinc-50">Sample {data.sample.sample_id}</span>
              <span className="text-zinc-400">{data.sample.split} split</span><span className="text-zinc-400">actual {data.sample.actual_label}</span>
              <span className="text-zinc-400">{data.sample.timestamp}</span><span className="text-zinc-400">{data.sample.n_model_features} features</span>
            </div>
            <div className="lg:col-span-2"><RiskCard p={data.prediction} split={data.sample.split} /></div>
            <AnomalyPanel anomaly={anomalyData ?? data.anomaly} />
            <div className="lg:col-span-3"><div className="mb-2 flex justify-end"><button onClick={refreshRootCause} className="btn-ghost text-xs" disabled={rootCauseBusy}>{rootCauseBusy ? "Refreshing root cause…" : "Refresh root-cause endpoint"}</button></div><Candidates items={data.root_cause_candidates} /></div>
            <div className="lg:col-span-3"><Shap pos={data.explanation.top_positive} neg={data.explanation.top_negative} /></div>
            <div className="lg:col-span-3">{top && data.sample.sample_id !== null && <WhatIfPanel sampleId={data.sample.sample_id} feature={top.feature} current={top.current_value} />}</div>
            <p className="text-xs leading-relaxed text-zinc-500 lg:col-span-3">{data.causality_note} {data.disclaimer}</p>
          </div>)}
      </div>
    </main>
  );
}
