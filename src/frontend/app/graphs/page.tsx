"use client";
import { useEffect, useState } from "react";
import { analyzeFull, getFeatureImportance, ApiError } from "@/lib/api/client";
import type { AnalyzeFull, FeatureImportance } from "@/lib/types/api";
import { DistributionChart, ImportanceChart, ImportanceLegend, ShapChart } from "@/components/charts/Charts";

export default function Graphs() {
  const [fi, setFi] = useState<FeatureImportance | null>(null);
  const [id, setId] = useState("100");
  const [d, setD] = useState<AnalyzeFull | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const msg = (e: unknown) => (e instanceof ApiError ? e.message : "Unexpected error.");
  useEffect(() => { getFeatureImportance(20).then(setFi).catch((e) => setErr(msg(e))); }, []);
  async function load() { setErr(null); try { setD(await analyzeFull(Number(id), 5)); } catch (e) { setErr(msg(e)); } }
  const card = "card p-6";
  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <h1 className="text-3xl font-bold">Graphs</h1>
      {err && <p role="alert" className="mt-4 text-risk">{err}</p>}
      <section className={`${card} mt-6 animate-slide-in !bg-panel/70 backdrop-blur-md`} aria-label="Global feature importance">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><h2 className="eyebrow !text-sm !text-zinc-200">Global Feature Importance (mean |SHAP|)</h2>
            <p className="mt-1 text-xs text-zinc-400/80">Shows which sensors have the greatest influence on the model’s defect prediction.</p></div>
          <ImportanceLegend /></div>
        <div className="mt-3">{fi ? <ImportanceChart fi={fi} /> : !err && <p className="mt-3 text-sm text-zinc-500">Loading…</p>}</div>
        <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">Longer bars indicate greater model influence. Color indicates relative importance: green = low, orange = medium, red = high. Red does not mean the physical sensor is defective.</p></section>
      <form className="mt-8 flex items-end gap-3" onSubmit={(e) => { e.preventDefault(); load(); }}>
        <label className="text-sm">Sample ID<input value={id} onChange={(e) => setId(e.target.value)} className="field w-40" /></label>
        <button className="btn-primary">Plot sample</button>
      </form>
      {d && (<div className="mt-6 grid gap-4 md:grid-cols-2">
        <section className={card}><h2 className="eyebrow">SHAP contributions, sample {d.sample.sample_id} (live)</h2>
          <ShapChart rows={[...d.explanation.top_positive, ...d.explanation.top_negative]} /></section>
        {d.root_cause_candidates.slice(0, 3).map((c) => (
          <section key={c.feature} className={card}><h2 className="eyebrow">{c.feature}: PASS vs FAIL vs this sample (live)</h2><DistributionChart c={c} /></section>))}
      </div>)}
    </main>
  );
}
