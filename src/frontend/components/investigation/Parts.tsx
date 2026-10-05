"use client";
import { useState } from "react";
import type { Candidate, Contribution } from "@/lib/types/api";
import { useCountUp } from "@/lib/hooks/useCountUp";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
export const Tag = ({ demo }: { demo?: boolean }) => (
  <span className={`ml-2 text-xs ${demo ? "text-warn" : "text-ok"}`}>{demo ? "Demo data" : "Live"}</span>
);

export function RiskCard({ p, split }: { p: { failure_probability: number; predicted_class: string; risk_level: string; decision_threshold: number }; split: string }) {
  const shown = useCountUp(p.failure_probability * 100);
  const color = p.risk_level.toUpperCase() === "HIGH" ? "text-risk" : p.risk_level.toUpperCase() === "MEDIUM" ? "text-warn" : "text-ok";
  return (
    <section className="card animate-rise p-6" aria-label="Failure risk">
      <h3 className="eyebrow">Failure risk<Tag /></h3>
      <div className={`num mt-2 text-6xl font-medium ${color}`}>{shown.toFixed(1)}%</div>
      <p className="mt-1 text-sm text-zinc-300">{p.risk_level} risk · predicted {p.predicted_class} · threshold {pct(p.decision_threshold)}</p>
      <div className="relative mt-4 h-2 rounded bg-line" role="img" aria-label={`Risk ${pct(p.failure_probability)}`}>
        <div className="h-2 rounded bg-risk" style={{ width: pct(p.failure_probability) }} />
        <div className="absolute top-[-3px] h-3.5 w-px bg-white" style={{ left: pct(p.decision_threshold) }} title="Decision threshold" />
      </div>
      {split === "train" && <p className="mt-4 text-sm text-warn">In-sample row: it was used in training, so risk may be optimistic. Prefer validation or test samples.</p>}
    </section>
  );
}

export function Shap({ pos, neg }: { pos: Contribution[]; neg: Contribution[] }) {
  const rows = [...pos, ...neg];
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9);
  return (
    <section className="card p-6" aria-label="Model explanation">
      <h3 className="eyebrow">Contribution to predicted failure (log-odds)<Tag /></h3>
      <ul className="mt-4 space-y-2">
        {rows.map((r) => (
          <li key={r.feature} className="grid grid-cols-[110px_1fr_60px] items-center gap-3 text-sm">
            <span>{r.feature}</span>
            <div className="h-2 rounded bg-line"><div className={`h-2 rounded ${r.contribution >= 0 ? "bg-risk" : "bg-ok"}`} style={{ width: `${(Math.abs(r.contribution) / max) * 100}%` }} /></div>
            <span className="text-right tabular-nums">{r.contribution >= 0 ? "+" : ""}{r.contribution.toFixed(3)}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-zinc-500">Red pushes toward FAIL. Green pushes toward PASS.</p>
    </section>
  );
}

export function Candidates({ items }: { items: Candidate[] }) {
  const [open, setOpen] = useState<string | null>(items[0]?.feature ?? null);
  if (!items.length) return <p className="card p-6 text-sm text-zinc-400">The backend returned no root-cause candidates for this sample.</p>;
  return (
    <section className="card p-6" aria-label="Root-cause candidates">
      <h3 className="eyebrow">Root-cause candidates<Tag /></h3>
      <p className="text-xs text-zinc-500">Model-derived hypotheses from SHAP, deviation, association and history.</p>
      <ol className="mt-4 divide-y divide-line">
        {items.map((c, i) => {
          const isOpen = open === c.feature;
          return (
            <li key={c.feature} className="py-3">
              <button className="flex w-full items-center gap-4 text-left" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : c.feature)}>
                <span className="w-6 text-zinc-500">{i + 1}</span>
                <span className="w-28 font-medium">{c.feature}</span>
                <span className="h-1.5 flex-1 rounded bg-line"><span className="block h-1.5 rounded bg-accent" style={{ width: pct(c.root_cause_score) }} /></span>
                <span className="w-14 text-right tabular-nums">{pct(c.root_cause_score)}</span>
                <span className="w-20 text-right text-xs text-zinc-400">{c.confidence}</span>
              </button>
              {isOpen && (
                <div className="mt-3 grid gap-4 pl-10 text-sm md:grid-cols-2">
                  <dl className="grid grid-cols-2 gap-y-1">
                    <dt className="text-zinc-500">Current value</dt><dd>{c.current_value}</dd>
                    <dt className="text-zinc-500">SHAP</dt><dd>{c.shap_contribution.toFixed(3)} (rank {c.shap_rank})</dd>
                    <dt className="text-zinc-500">Deviation</dt><dd>{c.deviation_sigma.toFixed(2)}σ</dd>
                    {c.normal_stats && <><dt className="text-zinc-500">PASS mean ± σ</dt><dd>{c.normal_stats.mean.toFixed(2)} ± {c.normal_stats.std.toFixed(2)}</dd></>}
                    {c.failure_stats && <><dt className="text-zinc-500">FAIL mean ± σ</dt><dd>{c.failure_stats.mean.toFixed(2)} ± {c.failure_stats.std.toFixed(2)}</dd></>}
                    {c.failure_association && <><dt className="text-zinc-500">Association</dt><dd>AUC {c.failure_association.auc.toFixed(2)}, q={c.failure_association.q_value.toExponential(1)}{c.failure_association.significant ? " (significant)" : ""}</dd></>}
                    {c.historical_evidence && <><dt className="text-zinc-500">History</dt><dd>{c.historical_evidence.abnormal_same_direction}/{c.historical_evidence.similar_fail_samples_checked} similar fails abnormal</dd></>}
                    <dt className="text-zinc-500">Evidence</dt><dd>{c.evidence_strength}</dd>
                  </dl>
                  <p className="text-zinc-300">{c.explanation}<span className="mt-2 block text-xs text-zinc-500">{c.label}</span></p>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
