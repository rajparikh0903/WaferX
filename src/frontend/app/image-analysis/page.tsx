"use client";
import { useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { analyzeWithImage, imageAnalyze, ApiError, getImageModelInfo } from "@/lib/api/client";
import type { CombinedAnalysis } from "@/lib/types/api";
import { RiskCard, Tag, Candidates } from "@/components/investigation/Parts";

export default function ImageAnalysis() {
  const [src, setSrc] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [id, setId] = useState("100");
  const [dataset, setDataset] = useState<"wm" | "mix">("wm");
  const [data, setData] = useState<CombinedAnalysis | null>(null);
  const [imageOnly, setImageOnly] = useState<import("@/lib/types/api").ImageAnalysisResult | null>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);

  async function take(f?: File) {
    if (!f) return;
    if (!f.type.startsWith("image/") && !/\.(npy|npz)$/i.test(f.name)) { setErr("Upload PNG/JPG/WebP or NPY/NPZ wafer-map data."); return; }
    setErr(null);
    if (src) URL.revokeObjectURL(src);
    setSrc(f.type.startsWith("image/") ? URL.createObjectURL(f) : null);
    setImageOnly(null);
    setName(f.name);
    setFile(f);
    try { setModelInfo(await getImageModelInfo(dataset)); } catch { setModelInfo(null); }
  }

  async function runImageOnly() {
    if (!file) { setErr("Choose a wafer map first."); return; }
    setBusy(true); setErr(null);
    try { setImageOnly(await imageAnalyze(file, dataset)); }
    catch (e) { setErr(e instanceof ApiError ? e.message : "Unexpected image-analysis error."); }
    finally { setBusy(false); }
  }

  async function run() {
    const n = Number(id);
    if (!file) { setErr("Choose a wafer map first."); return; }
    if (!Number.isInteger(n) || n < 0) { setErr("Sample ID must be a whole number of 0 or more."); return; }
    setBusy(true); setErr(null);
    try { setData(await analyzeWithImage(file, n, dataset, 5)); }
    catch (e) { setErr(e instanceof ApiError ? e.message : "Unexpected error."); setData(null); }
    finally { setBusy(false); }
  }

  const img = data?.wafer_image_intelligence;
  const process = data?.process_intelligence;
  const top3 = img ? Object.entries(img.top3) : [];

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <h1 className="text-3xl font-bold">Defect image analysis</h1>
      <p className="mt-1 text-sm text-zinc-400">The uploaded wafer map is sent to the real DINOv2 + ViT embedding pipeline and the linked SECOM sample is sent through the process/root-cause engine.</p>
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <label className="text-sm">Linked sample ID<input value={id} onChange={(e) => setId(e.target.value)} className="field mt-1 w-full" /></label>
        <label className="text-sm">Image model family
          <select value={dataset} onChange={async (e) => { const next = e.target.value as "wm" | "mix"; setDataset(next); try { setModelInfo(await getImageModelInfo(next)); } catch { setModelInfo(null); } }} className="mt-1 block w-full rounded-md border border-line bg-bg px-3 py-2 text-white">
            <option value="wm">WM — wafer-map pattern classifier</option><option value="mix">MIX — binary-code classifier</option>
          </select>
        </label>
        <div className="self-end text-xs text-zinc-500">{modelInfo ? `${modelInfo.classes?.length ?? 0} classes · ${modelInfo.embedding_tags?.join(" + ")}` : "Model metadata loads from GET /image/model-info."}</div>
      </div>

      <div role="button" tabIndex={0} onClick={() => input.current?.click()} onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
        onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); take(e.dataTransfer.files[0]); }}
        className="mt-4 flex min-h-40 cursor-pointer items-center justify-center rounded-xl border border-dashed border-line bg-panel text-center text-sm text-zinc-400">
        {file ? <span><strong className="text-zinc-200">{name}</strong><br />Click or drop to replace it</span> : <span>Drop a wafer image here, or press to choose a file</span>}
        <input ref={input} type="file" accept="image/*,.npy,.npz" hidden onChange={(e) => take(e.target.files?.[0])} />
      </div>
      {file && <div className="mt-4 flex flex-wrap gap-2">
        <button onClick={runImageOnly} className="btn-ghost" disabled={busy}>{busy ? <><Loader2 size={15} className="animate-spin" />Running…</> : "Image-only analysis"}</button>
        <button onClick={run} className="btn-primary" disabled={busy}>{busy ? <><Loader2 size={15} className="animate-spin" />Running image + process intelligence…</> : "Analyze wafer + linked sample"}</button>
      </div>}
      {err && <p role="alert" className="mt-4 text-risk">{err}</p>}

      {imageOnly && <section className="card mt-6 p-6"><h2 className="eyebrow">Standalone image endpoint · POST /image/analyze<Tag /></h2><div className="mt-3 grid gap-3 sm:grid-cols-4 text-sm"><div><span className="text-zinc-500">Class</span><p className="font-semibold">{imageOnly.label}</p></div><div><span className="text-zinc-500">Confidence</span><p className="font-semibold">{pct(imageOnly.confidence)}</p></div><div><span className="text-zinc-500">Anomaly</span><p className={`font-semibold ${imageOnly.anomaly_label ? "text-risk" : "text-ok"}`}>{imageOnly.anomaly_label ? "ANOMALOUS" : "NORMAL"}</p></div><div><span className="text-zinc-500">Detector</span><p className="font-semibold">{imageOnly.detector_score.toFixed(3)}</p></div></div></section>}

      {data && img && process && (
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <section className="card p-4">
            <h2 className="eyebrow">Uploaded image<Tag /></h2>
            {src ? <img src={src} alt={`Wafer map ${name}`} className="mt-3 w-full rounded" /> : <div className="mt-3 grid aspect-square place-items-center rounded bg-line text-xs text-zinc-500">NPY/NPZ map uploaded</div>}
            <p className="mt-3 text-xs text-zinc-500">Map shape: {img.input?.map_shape?.join(" × ") ?? "N/A"}</p>
          </section>
          <section className="card p-6">
            <h2 className="eyebrow">Defect classification<Tag /></h2>
            <p className="mt-2 text-2xl font-semibold">{img.label}</p>
            <p className="text-sm text-zinc-400">Confidence {pct(img.confidence)} · dataset {img.dataset.toUpperCase()}</p>
            <ul className="mt-4 space-y-2 text-sm">
              {top3.map(([k, p]) => <li key={k} className="grid grid-cols-[1fr_1fr_50px] items-center gap-3"><span>{k}</span><div className="h-1.5 rounded bg-line"><div className="h-1.5 rounded bg-accent" style={{ width: `${p * 100}%` }} /></div><span className="text-right tabular-nums">{pct(p)}</span></li>)}
            </ul>
            <h2 className="mt-6 eyebrow">Image anomaly<Tag /></h2>
            <p className={`mt-2 text-xl font-semibold ${img.anomaly_label ? "text-risk" : "text-ok"}`}>{img.anomaly_label ? "ANOMALOUS" : "NORMAL"}</p>
            <p className="text-sm text-zinc-400">Classifier anomaly {pct(img.anomaly_score_classifier)} · detector score {img.detector_score.toFixed(3)}</p>
          </section>
          <RiskCard p={process.prediction} split={process.sample.split} />
        </div>
      )}
      {process && <div className="mt-4"><h2 className="mb-2 text-sm text-zinc-400">Associated process root-cause evidence for sample {process.sample.sample_id}</h2><Candidates items={process.root_cause_candidates} /></div>}
      {data?.integration && <p className="mt-4 text-xs text-zinc-500">Integrated models: {data.integration.process_model} · {data.integration.image_model}</p>}
    </main>
  );
}

function pct(x: number) { return `${(x * 100).toFixed(1)}%`; }
