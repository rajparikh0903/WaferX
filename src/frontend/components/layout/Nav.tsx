"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X, Activity } from "lucide-react";
import { useBackendStatus } from "@/lib/hooks/useBackendStatus";

const links = [["/", "Overview"], ["/investigate", "Investigate"], ["/graphs", "Graphs"], ["/what-if", "What-if"], ["/assistant", "AI Engineer Assistant"], ["/image-analysis", "Images"], ["/models", "Models"]];
const dot = { checking: "bg-zinc-500", online: "bg-ok", offline: "bg-risk" } as const;
const label = { checking: "Checking engine", online: "Root-cause engine online", offline: "Root-cause engine offline" } as const;

export default function Nav() {
  const path = usePathname(); const [open, setOpen] = useState(false); const st = useBackendStatus();
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:rounded focus:bg-accent focus:px-3 focus:py-1 focus:text-black">Skip to content</a>
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold text-zinc-50"><Activity size={18} className="text-accent" aria-hidden />Yield Intelligence</Link>
        <nav aria-label="Primary" className="hidden gap-1 md:flex">
          {links.map(([h, l]) => <Link key={h} href={h} aria-current={path === h ? "page" : undefined}
            className={`rounded-md px-3 py-1.5 text-sm transition ${path === h ? "bg-raised text-white" : "text-zinc-400 hover:text-white"}`}>{l}</Link>)}
        </nav>
        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-2 text-xs text-zinc-400 sm:flex" role="status"><span className={`h-2 w-2 rounded-full ${dot[st]}`} />{label[st]}</span>
          <button className="btn-ghost md:hidden" aria-label="Toggle menu" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? <X size={16} /> : <Menu size={16} />}</button>
        </div>
      </div>
      {open && <nav aria-label="Mobile" className="grid gap-1 border-t border-line p-3 md:hidden">
        {links.map(([h, l]) => <Link key={h} href={h} onClick={() => setOpen(false)} className={`rounded-md px-3 py-2 text-sm ${path === h ? "bg-raised text-white" : "text-zinc-400"}`}>{l}</Link>)}
        <span className="px-3 pt-2 text-xs text-zinc-500">{label[st]}</span></nav>}
    </header>
  );
}
