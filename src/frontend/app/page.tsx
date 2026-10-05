import Link from "next/link";
import { ArrowRight } from "lucide-react";
const loop = [
  ["Detect", "Anomaly detection", "Flags lots that drift from normal before they fail.", "live", "/investigate"],
  ["Classify", "Defect images", "Classifies wafer-map patterns and image anomalies.", "live", "/image-analysis"],
  ["Explain", "Root-cause candidates", "Ranks sensors by SHAP, deviation and history.", "live", "/investigate"],
  ["Simulate", "What-if", "Tests how a parameter change moves failure risk.", "live", "/what-if"],
] as const;
export default function Home() {
  return (
    <main className="mx-auto max-w-6xl px-4 pb-12 pt-14 sm:px-6 sm:pt-20">
      <p className="eyebrow">IBM × VGEC Semiconductor AI Hackathon</p>
      <h1 className="mt-4 max-w-4xl text-4xl font-semibold leading-[1.05] sm:text-6xl">Find the cause. Change one parameter. Keep what you learn.</h1>
      <p className="mt-5 max-w-xl text-base text-zinc-400 sm:text-lg">Yield Intelligence turns sensor signals into one ranked answer, then a tested change.</p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link href="/investigate" className="btn-primary">Open an investigation <ArrowRight size={15} /></Link>
        <Link href="/graphs" className="btn-ghost">View graphs</Link>
      </div>
      <h2 className="mt-20 text-xl font-semibold sm:text-2xl">One loop, from detection to correction.</h2>
      <ol className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {loop.map(([k, t, d, s, href], i) => (
          <li key={t}><Link href={href} className="card group block h-full p-5 transition hover:border-zinc-600">
            <span className="num text-xs text-zinc-500">{i + 1} · {k}</span>
            <span className="mt-2 block text-base font-semibold text-zinc-50">{t}</span>
            <span className="mt-2 block text-sm text-zinc-400">{d}</span>
            <span className={`mt-4 inline-flex items-center gap-1.5 text-xs ${s === "live" ? "text-ok" : "text-warn"}`}><span className="h-1.5 w-1.5 rounded-full bg-current" />{s === "live" ? "Live backend" : "Demo module"}</span>
          </Link></li>))}
      </ol>
    </main>
  );
}
