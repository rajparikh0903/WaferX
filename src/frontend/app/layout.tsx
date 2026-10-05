import "./globals.css";
import { Inter, IBM_Plex_Mono } from "next/font/google";
import Nav from "@/components/layout/Nav";
const sans = Inter({ subsets: ["latin"], variable: "--font-sans" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });
export const metadata = { title: "Yield Intelligence · IBM × VGEC", description: "Root-cause intelligence for semiconductor yield." };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en" className={`${sans.variable} ${mono.variable}`}><body><Nav /><div id="main">{children}</div>
    <footer className="mx-auto mt-16 max-w-6xl border-t border-line px-4 py-6 text-xs text-zinc-500 sm:px-6">IBM × VGEC Semiconductor AI Hackathon. Root-cause outputs are model-based hypotheses, not confirmed physical causes.</footer></body></html>);
}
