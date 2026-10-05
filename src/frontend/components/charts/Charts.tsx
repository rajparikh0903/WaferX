"use client";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Candidate, Contribution, FeatureImportance } from "@/lib/types/api";
const axis = { stroke: "#8a8f98", fontSize: 12 };
const tip = { contentStyle: { background: "#111214", border: "1px solid #23262b" } };

const STOPS: [number, number, number][] = [[53, 196, 106], [242, 169, 59], [240, 80, 74]]; // green, orange, red
export function importanceColor(t: number) {
  const x = Math.min(Math.max(t, 0), 1) * 2, k = Math.min(Math.floor(x), 1), f = x - k;
  const c = STOPS[k].map((a, n) => Math.round(a + (STOPS[k + 1][n] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
export function ImportanceChart({ fi }: { fi: FeatureImportance }) {
  const max = Math.max(...fi.features.map((f) => f.mean_abs_shap), 1e-12); // relative importance = value / max
  return (<ResponsiveContainer width="100%" height={Math.max(260, fi.features.length * 24)}>
    <BarChart data={fi.features} layout="vertical" margin={{ left: 30 }}>
      <CartesianGrid stroke="#23262b" horizontal={false} /><XAxis type="number" {...axis} /><YAxis type="category" dataKey="feature" width={90} {...axis} />
      <Tooltip {...tip} cursor={{ fill: "rgba(255,255,255,.04)" }} />
      <Bar dataKey="mean_abs_shap" name="mean |SHAP|" radius={[0, 4, 4, 0]} isAnimationActive animationDuration={700}>
        {fi.features.map((f) => <Cell key={f.feature} fill={importanceColor(f.mean_abs_shap / max)} fillOpacity={0.8} />)}
      </Bar>
    </BarChart></ResponsiveContainer>);
}
export function ImportanceLegend() {
  return (<ul className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-zinc-400" aria-label="Color legend">
    {[["Low importance", importanceColor(0)], ["Medium importance", importanceColor(0.5)], ["High importance", importanceColor(1)]].map(([l, c]) =>
      <li key={l} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: c, opacity: 0.85 }} />{l}</li>)}</ul>);
}
export function ShapChart({ rows }: { rows: Contribution[] }) {
  return (<ResponsiveContainer width="100%" height={Math.max(220, rows.length * 28)}>
    <BarChart data={rows} layout="vertical" margin={{ left: 30 }}>
      <CartesianGrid stroke="#23262b" horizontal={false} /><XAxis type="number" {...axis} /><YAxis type="category" dataKey="feature" width={90} {...axis} />
      <Tooltip {...tip} /><Bar dataKey="contribution">{rows.map((r) => <Cell key={r.feature} fill={r.contribution >= 0 ? "#f0443a" : "#3fc95a"} />)}</Bar>
    </BarChart></ResponsiveContainer>);
}
// PASS mean vs FAIL mean vs this sample, per candidate, in σ of the PASS distribution (all values from the backend).
export function DistributionChart({ c }: { c: Candidate }) {
  if (!c.normal_stats || !c.failure_stats) return <p className="text-sm text-zinc-500">No distribution stats returned for {c.feature}.</p>;
  const s = c.normal_stats.std || 1;
  const z = (v: number) => +((v - c.normal_stats!.mean) / s).toFixed(2);
  const data = [{ k: "PASS mean", v: 0 }, { k: "FAIL mean", v: z(c.failure_stats.mean) }, { k: "This sample", v: z(c.current_value) }];
  return (<ResponsiveContainer width="100%" height={200}><BarChart data={data}>
    <CartesianGrid stroke="#23262b" vertical={false} /><XAxis dataKey="k" {...axis} /><YAxis {...axis} unit="σ" /><Tooltip {...tip} />
    <Bar dataKey="v" name="σ from PASS mean">{data.map((d, i) => <Cell key={i} fill={["#3fc95a", "#f5a524", "#f0443a"][i]} />)}</Bar></BarChart></ResponsiveContainer>);
}
export function WhatIfCurve({ points, current }: { points: { value: number; risk: number }[]; current?: number }) {
  return (<ResponsiveContainer width="100%" height={260}><LineChart data={points}>
    <CartesianGrid stroke="#23262b" /><XAxis dataKey="value" type="number" domain={["dataMin", "dataMax"]} {...axis} />
    <YAxis {...axis} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} /><Tooltip {...tip} /><Legend />
    <Line dataKey="risk" name="Predicted failure risk" stroke="#5b9cff" dot />
    {current !== undefined && <ReferenceLine x={current} stroke="#f0443a" label={{ value: "current", fill: "#f0443a", fontSize: 12 }} />}
  </LineChart></ResponsiveContainer>);
}
