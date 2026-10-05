"use client";
import { useEffect, useRef, useState } from "react";
import { analyzeFull, whatIf, ApiError } from "@/lib/api/client";
import type { AnalyzeFull } from "@/lib/types/api";

// Read a number from a response object by trying several likely key names (shape is not documented).
const pick = (o: any, keys: string[]) => { for (const k of keys) if (typeof o?.[k] === "number") return o[k] as number; return undefined; };
const pct = (x?: number) => (x === undefined ? "N/A" : `${(x * 100).toFixed(1)}%`);

export default function WhatIfPage() {
  const [id, setId] = useState("100");
  const [base, setBase] = useState<AnalyzeFull | null>(null);
  const [feature, setFeature] = useState("");
  const [value, setValue] = useState<number | null>(null);
  const [res, setRes] = useState<Record<string, any> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ctl = useRef<AbortController | null>(null);
  const msg = (e: unknown) => (e instanceof ApiError ? e.message : "Unexpected error.");

  async function load() {
    setErr(null); setRes(null);
    try { const a = await analyzeFull(Number(id), 5); setBase(a); const c = a.root_cause_candidates[0]; if (c) { setFeature(c.feature); setValue(c.current_value); } }
    catch (e) { setErr(msg(e)); }
  }
  const cand = base?.root_cause_candidates.find((c) => c.feature === feature);
  const lo = cand?.normal_stats ? Math.min(cand.normal_stats.p5, cand.current_value) : 0;
  const hi = cand?.failure_stats ? Math.max(cand.failure_stats.p95, cand.current_value) : 1;

  useEffect(() => { // debounced, cancellable backend call
    if (!base || !feature || value === null) return;

    // Capture narrowed values before entering the async timeout callback.
    // This keeps the existing runtime behavior while satisfying the production TypeScript build.
    const sampleId = base.sample.sample_id;
    const testValue = value;

    const t = setTimeout(async () => {
      ctl.current?.abort();
      const controller = new AbortController();
      ctl.current = controller;
      setBusy(true);
      setErr(null);

      try {
        setRes(await whatIf(sampleId, feature, testValue, controller.signal));
      } catch (e: any) {
        if (e?.name !== "AbortError") setErr(msg(e));
      }
      setBusy(false);
    }, 400);

    return () => clearTimeout(t);
  }, [base, feature, value]);

  const before = pick(res, ["current_risk", "baseline_probability", "baseline_risk", "original_probability"]) ?? base?.prediction.failure_probability;
  const after = pick(res, ["scenario_risk", "new_probability", "scenario_probability", "counterfactual_probability", "failure_probability"]);
  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <h1 className="text-3xl font-bold">What-if simulator</h1>
      <p className="mt-1 text-sm text-zinc-400">The selected sensor and curve are evaluated by the FastAPI POST /what-if endpoint.</p>
      <form className="mt-6 flex items-end gap-3" onSubmit={(e) => { e.preventDefault(); load(); }}>
        <label className="text-sm">Sample ID<input value={id} onChange={(e) => setId(e.target.value)} className="field w-40" /></label>
        <button className="btn-primary">Load sample</button>
      </form>
      {err && <p role="alert" className="mt-4 text-risk">{err}</p>}
      {base && cand && value !== null && (
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <section className="card p-6">
            <label className="eyebrow">Parameter
              <select value={feature} onChange={(e) => { setFeature(e.target.value); const c = base.root_cause_candidates.find((x) => x.feature === e.target.value); if (c) setValue(c.current_value); }} className="mt-1 block w-full rounded-md border border-line bg-bg px-3 py-2 text-white">
                {base.root_cause_candidates.map((c) => <option key={c.feature}>{c.feature}</option>)}</select></label>
            <p className="mt-4 text-sm">Current {cand.current_value} → test <span className="font-semibold">{value.toFixed(3)}</span></p>
            <input type="range" aria-label="Test value" className="mt-2 w-full" min={lo} max={hi} step={(hi - lo) / 200} value={value} onChange={(e) => setValue(Number(e.target.value))} />
          </section>
          <section className="card p-6" aria-live="polite">
            <p className="text-xs text-ok">Live · POST /what-if{busy && " · updating…"}</p>
            <p className="mt-2 text-3xl font-semibold">{pct(before)} → {pct(after)}</p>
            <p className="eyebrow">Change: {before !== undefined && after !== undefined ? `${((after - before) * 100).toFixed(1)} percentage points` : "N/A"}</p>
            <p className="mt-4 text-xs text-warn">Model-based counterfactual. Requires engineering validation before any physical process change.</p>
          </section>
          {res && <details className="card p-6 md:col-span-2"><summary className="cursor-pointer text-sm text-zinc-400">Raw /what-if response</summary><pre className="mt-3 overflow-x-auto text-xs">{JSON.stringify(res, null, 2)}</pre></details>}
          {base.recommendation && Object.keys(base.recommendation).length > 0 && <details open className="card p-6 md:col-span-2"><summary className="cursor-pointer text-sm text-zinc-400">Backend recommendation (from /analyze)</summary><pre className="mt-3 overflow-x-auto text-xs">{JSON.stringify(base.recommendation, null, 2)}</pre></details>}
        </div>)}
    </main>
  );
}
