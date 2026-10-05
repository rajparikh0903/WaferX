import { Check, Loader2 } from "lucide-react";
export type Step = { label: string; source: "live" | "demo" };
export default function Stepper({ steps, done, running }: { steps: Step[]; done: number; running: boolean }) {
  return (
    <ol className="card grid grid-cols-2 gap-px overflow-hidden bg-line p-0 sm:grid-cols-3 lg:grid-cols-6" aria-label="Investigation progress">
      {steps.map((s, i) => {
        const state = i < done ? "done" : running && i === done ? "run" : "idle";
        return (<li key={s.label} className="flex items-center gap-3 bg-panel px-4 py-3">
          <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-xs ${state === "done" ? "bg-ok/15 text-ok" : state === "run" ? "bg-accent/15 text-accent" : "bg-raised text-zinc-500"}`}>
            {state === "done" ? <Check size={13} /> : state === "run" ? <Loader2 size={13} className="animate-spin" /> : i + 1}</span>
          <span><span className="block text-sm text-zinc-100">{s.label}</span>
            <span className={`text-[11px] ${s.source === "live" ? "text-ok" : "text-warn"}`}>{s.source === "live" ? "Live" : "Demo"}</span></span></li>);
      })}
    </ol>
  );
}
