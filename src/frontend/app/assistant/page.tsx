import AssistantChat from "@/components/assistant/AssistantChat";
export const metadata = { title: "AI Engineer Assistant · Yield Intelligence" };
export default function Assistant() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <header className="mb-5"><h1 className="text-2xl font-semibold sm:text-3xl">YieldTwin AI Engineer Assistant</h1>
        <p className="mt-1 text-sm text-zinc-400">AI-powered wafer failure investigation &amp; engineering support</p></header>
      <AssistantChat />
      <p className="mt-3 text-xs text-zinc-500">Root-cause outputs are model-based hypotheses. What-if results are model-based counterfactuals and require engineering validation.</p>
    </main>
  );
}
