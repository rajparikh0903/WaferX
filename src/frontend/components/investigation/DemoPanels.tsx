"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { whatIf } from "@/lib/api/client";
import type { AnomalyInfo } from "@/lib/types/api";
import { Tag } from "./Parts";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export function AnomalyPanel({ anomaly }: { anomaly?: AnomalyInfo }) {
  if (!anomaly) return <section className="card p-6"><h3 className="eyebrow">Anomaly status<Tag /></h3><p className="mt-2 text-sm text-zinc-400">No anomaly result returned.</p></section>;
  const anomalous = Boolean(anomaly.anomaly_label);
  return (
    <section className="card p-6" aria-label="Anomaly status">
      <h3 className="eyebrow">Anomaly status<Tag /></h3>
      <div className={`mt-2 text-2xl font-semibold ${anomalous ? "text-risk" : "text-ok"}`}>{anomalous ? "ANOMALOUS" : "NORMAL"}</div>
      <p className="eyebrow">Score {typeof anomaly.effective_score === "number" ? pct(anomaly.effective_score) : "N/A"} · severity {anomaly.severity ?? "N/A"}</p>
      <p className="mt-3 text-xs text-zinc-500">{anomaly.source ?? anomaly.model ?? "Isolation Forest"}</p>
    </section>
  );
}

export function WhatIfPanel({ sampleId, feature, current }: { sampleId: number; feature: string; current: number }) {
  const [res, setRes] = useState<Record<string, any> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setBusy(true); setErr(null);
    const delta = Math.max(Math.abs(current) * 0.02, 0.01);
    const values = [current - delta, current, current + delta];
    whatIf(sampleId, feature, values).then((data) => { if (alive) setRes(data); }).catch((e) => { if (alive) setErr(e instanceof Error ? e.message : "Unable to load what-if result."); }).finally(() => { if (alive) setBusy(false); });
    return () => { alive = false; };
  }, [sampleId, feature, current]);

  return (
    <section className="card p-6" aria-label="What-if simulator">
      <div className="flex items-start justify-between gap-3">
        <div><h3 className="eyebrow">Live what-if check<Tag /></h3><p className="text-xs text-zinc-500">Backend prediction for {feature}; open the full simulator for arbitrary values.</p></div>
        <Link href="/what-if" className="btn-ghost text-xs">Full simulator</Link>
      </div>
      {busy && <p className="mt-4 flex items-center gap-2 text-sm text-zinc-400"><Loader2 size={14} className="animate-spin" />Querying POST /what-if…</p>}
      {err && <p className="mt-4 text-sm text-risk">{err}</p>}
      {res && !busy && (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div><p className="text-xs text-zinc-500">Current value</p><p className="text-lg font-semibold">{Number(res.current_value).toFixed(4)}</p></div>
          <div><p className="text-xs text-zinc-500">Current failure risk</p><p className="text-lg font-semibold">{pct(res.current_risk)}</p></div>
          <div><p className="text-xs text-zinc-500">Backend curve points</p><p className="text-lg font-semibold">{Array.isArray(res.curve) ? res.curve.length : 0}</p></div>
        </div>
      )}
      <p className="mt-3 text-xs text-warn">Model-based counterfactual; engineering validation is required before physical process changes.</p>
    </section>
  );
}
