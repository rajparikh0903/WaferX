"use client";
import { useEffect, useState } from "react";
import { getAnomalyInfo, getFeatureImportance, getImageModelInfo, getModelInfo, health, ApiError } from "@/lib/api/client";
import type { FeatureImportance, HealthResponse, ImageModelInfo, ModelInfo } from "@/lib/types/api";

export default function Models() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [fi, setFi] = useState<FeatureImportance | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [backend, setBackend] = useState<HealthResponse | null>(null);
  const [anomaly, setAnomaly] = useState<Record<string, any> | null>(null);
  const [imageWm, setImageWm] = useState<ImageModelInfo | null>(null);
  const [imageMix, setImageMix] = useState<ImageModelInfo | null>(null);
  useEffect(() => {
    Promise.all([getModelInfo(), getFeatureImportance(20), health(), getAnomalyInfo(), getImageModelInfo("wm"), getImageModelInfo("mix")]).then(([a, b, h, an, wm, mix]) => {
      setInfo(a); setFi(b); setBackend(h); setAnomaly(an); setImageWm(wm); setImageMix(mix);
    }).catch((e) => setErr(e instanceof ApiError ? e.message : "Unexpected error."));
  }, []);
  const max = Math.max(...(fi?.features.map((f) => f.mean_abs_shap) ?? [1]));
  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <h1 className="text-3xl font-bold">Model performance</h1>
      {err && <p role="alert" className="mt-6 text-risk">{err}</p>}
      {!info && !err && <p className="mt-6 text-zinc-400">Loading model metadata…</p>}
      {backend && <section className="card mt-6 p-6"><h2 className="eyebrow">Backend service health · GET /health</h2><div className="mt-3 grid gap-3 sm:grid-cols-4 text-sm"><div><span className="text-zinc-500">Root cause</span><p className="font-semibold text-ok">{backend.root_cause}</p></div><div><span className="text-zinc-500">Anomaly</span><p className="font-semibold text-ok">{backend.anomaly_detection}</p></div><div><span className="text-zinc-500">Image intelligence</span><p className="font-semibold text-ok">{backend.image_intelligence}</p></div><div><span className="text-zinc-500">Datasets</span><p className="font-semibold">{backend.image_datasets.join(", ") || "none"}</p></div></div></section>}
      {anomaly && <section className="card mt-4 p-6"><h2 className="eyebrow">Anomaly model · GET /anomaly-info</h2><pre className="mt-3 max-h-56 overflow-auto text-xs text-zinc-400">{JSON.stringify(anomaly, null, 2)}</pre></section>}
      {(imageWm || imageMix) && <section className="card mt-4 p-6"><h2 className="eyebrow">Wafer image models · GET /image/model-info</h2><div className="mt-3 grid gap-4 md:grid-cols-2">{[imageWm, imageMix].filter(Boolean).map((m) => <div key={m!.dataset} className="rounded-lg border border-line p-4"><p className="font-semibold">{m!.dataset.toUpperCase()}</p><p className="mt-1 text-xs text-zinc-500">{m!.classes.length} classes · {m!.embedding_tags.join(" + ")} · {m!.in_dim}-dim fused input</p><p className="mt-2 text-xs text-zinc-500">Normal class: {m!.normal_class}</p></div>)}</div></section>}
      {info && (
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <section className="card p-6">
            <h2 className="eyebrow">Model · {String(info.model ?? "N/A")} · v{String(info.model_version ?? "N/A")}</h2>
            <dl className="mt-3 grid grid-cols-2 gap-y-1 text-sm">
              {Object.entries(info.metrics ?? {}).filter(([, v]) => typeof v !== "object").map(([k, v]) => (<><dt key={k} className="text-zinc-500">{k}</dt><dd key={k + "v"}>{typeof v === "number" ? v.toFixed(3) : String(v)}</dd></>))}
            </dl>
            <p className="mt-4 text-xs text-zinc-500">{info.causality_note}</p>
          </section>
          <section className="card p-6">
            <h2 className="eyebrow">Global feature importance (mean |SHAP|)</h2>
            <ul className="mt-3 space-y-1.5 text-sm">
              {fi?.features.map((f) => (
                <li key={f.feature} className="grid grid-cols-[100px_1fr_50px] items-center gap-3">
                  <span>{f.feature}</span><div className="h-1.5 rounded bg-line"><div className="h-1.5 rounded bg-accent" style={{ width: `${(f.mean_abs_shap / max) * 100}%` }} /></div>
                  <span className="text-right tabular-nums">{f.mean_abs_shap.toFixed(3)}</span>
                </li>))}
            </ul>
          </section>
        </div>
      )}
    </main>
  );
}
