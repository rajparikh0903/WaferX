"use client";
import { useEffect, useRef, useState } from "react";
import { Send, Trash2, Bot, User, ChevronDown } from "lucide-react";
import { getEngineerAssistantResponse } from "@/lib/assistant/service";
import { FAQ } from "@/lib/assistant/faq";
import type { AssistantReply } from "@/lib/assistant/types";
import { ApiError } from "@/lib/api/client";

type Msg = { id: number; role: "engineer" | "assistant"; text: string; reply?: AssistantReply; error?: boolean };
let uid = 0;

export default function AssistantChat() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sample, setSample] = useState("100");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, busy]);

  async function ask(q: string) {
    q = q.trim(); if (!q || busy) return;
    const n = sample.trim() === "" ? NaN : Number(sample);
    setMsgs((m) => [...m, { id: ++uid, role: "engineer", text: q }]); setInput(""); setBusy(true);
    try {
      const reply = await getEngineerAssistantResponse(q, { sampleId: Number.isInteger(n) && n >= 0 ? n : null });
      setMsgs((m) => [...m, { id: ++uid, role: "assistant", text: reply.text, reply }]);
    } catch (e) {
      const text = e instanceof ApiError ? (e.kind === "offline" ? `Root-cause engine offline. ${e.message}` : e.message) : "Something went wrong.";
      setMsgs((m) => [...m, { id: ++uid, role: "assistant", text, error: true }]);
    }
    setBusy(false);
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <section className="card flex h-[calc(100vh-15rem)] min-h-[480px] flex-col overflow-hidden" aria-label="Chat">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
          <label className="flex items-center gap-2 text-xs text-zinc-400">Wafer context · sample ID
            <input value={sample} onChange={(e) => setSample(e.target.value)} inputMode="numeric" className="field num !mt-0 w-24 !py-1" aria-label="Sample ID" /></label>
          <button className="btn-ghost" onClick={() => setMsgs([])} disabled={!msgs.length}><Trash2 size={14} />Clear chat</button>
        </div>
        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4" role="log" aria-live="polite">
          {!msgs.length && <div className="grid h-full place-items-center text-center"><div><Bot className="mx-auto text-accent" size={28} /><p className="mt-3 text-zinc-200">Ask about a wafer</p><p className="mt-1 max-w-sm text-sm text-zinc-500">Answers come from the live backend for the sample ID above. Pick a quick question or type your own.</p></div></div>}
          {msgs.map((m) => m.role === "engineer" ? (
            <div key={m.id} className="flex animate-rise justify-end gap-2"><p className="max-w-[80%] rounded-lg rounded-tr-sm border border-accent/30 bg-accent/10 px-3 py-2 text-sm text-zinc-100">{m.text}</p><User size={16} className="mt-2 shrink-0 text-zinc-500" aria-hidden /></div>
          ) : (
            <div key={m.id} className="flex animate-rise gap-2"><Bot size={16} className="mt-2 shrink-0 text-accent" aria-hidden />
              <div className={`max-w-[85%] rounded-lg rounded-tl-sm border px-3 py-2 text-sm leading-relaxed ${m.error ? "border-risk/40 text-risk" : "border-line bg-raised text-zinc-200"}`}>
                <p className="whitespace-pre-line">{m.text}</p>
                {m.reply && <p className="mt-2 flex flex-wrap gap-2 text-[11px] text-zinc-500">
                  <span className={m.reply.kind === "live" ? "text-ok" : "text-zinc-400"}>{m.reply.kind === "live" ? "Live backend" : "Explainer"}</span>
                  {m.reply.sources.map((s) => <code key={s} className="num">{s}</code>)}</p>}
                {m.reply?.details && <details className="mt-2 text-xs"><summary className="cursor-pointer text-zinc-400">{m.reply.details.title}</summary><pre className="mt-2 max-h-56 overflow-auto rounded bg-bg p-2 text-[11px]">{JSON.stringify(m.reply.details.data, null, 2)}</pre></details>}
              </div></div>))}
          {busy && <div className="flex gap-2" role="status"><Bot size={16} className="mt-2 text-accent" /><p className="rounded-lg border border-line bg-raised px-3 py-2 text-sm text-zinc-400">Querying backend<span className="animate-pulse">…</span></p></div>}
          <div ref={end} />
        </div>
        <form className="flex gap-2 border-t border-line p-3" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about this wafer…" aria-label="Message" className="field !mt-0" />
          <button className="btn-primary" disabled={busy || !input.trim()} aria-label="Send"><Send size={15} /></button>
        </form>
      </section>
      <aside className="space-y-2" aria-label="Quick questions">
        <h2 className="eyebrow">FAQ / Quick questions</h2>
        {FAQ.map((g, i) => (
          <details key={g.group} open={i === 0} className="card group px-3 py-2">
            <summary className="flex cursor-pointer items-center justify-between text-sm font-medium text-zinc-100">{g.group}<ChevronDown size={14} className="transition group-open:rotate-180" /></summary>
            <div className="mt-2 flex flex-col gap-1">{g.items.map((q) => <button key={q} onClick={() => ask(q)} disabled={busy} className="rounded-md px-2 py-1.5 text-left text-[13px] text-zinc-300 transition hover:bg-raised hover:text-white disabled:opacity-50">{q}</button>)}</div>
          </details>))}
      </aside>
    </div>
  );
}
