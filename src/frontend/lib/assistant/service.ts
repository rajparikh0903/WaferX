// ============================================================================
// ASSISTANT SERVICE LAYER — the only place that maps questions to backend calls
// and backend responses to text. The chat UI imports getEngineerAssistantResponse()
// and nothing else from here. Adjust request/response mapping in this file only.
// Endpoints used (existing): /model-info, /feature-importance, /predict, /analyze, /what-if.
// ============================================================================
import { analyzeFull, getFeatureImportance, getModelInfo, predict, whatIf } from "@/lib/api/client";
import type { AnalyzeFull, Candidate } from "@/lib/types/api";
import type { AssistantReply, WaferContext } from "./types";
import { GRAPH_TEXT, RED_TEXT, RISK_TEXT, SHAP_TEXT } from "./knowledge";

type Intent = "whatif" | "shap" | "risk_meaning" | "graph" | "red" | "model" | "defect" | "history" | "deviation" | "first" | "top_sensor" | "why" | "predict" | "unknown";
const RULES: [Intent, RegExp][] = [
  ["whatif", /what.?if|chang|reduc|lower|increase|decrease/i],
  ["shap", /what does shap|shap mean/i],
  ["risk_meaning", /failure risk mean|what is failure risk/i],
  ["red", /shown in red|why.*red/i],
  ["graph", /importance graph|feature.?importance/i],
  ["model", /calculated|model|version|metric|roc|auc/i],
  ["defect", /defect|pattern|image/i],
  ["history", /similar|histor|before/i],
  ["deviation", /abnormal|normal|different/i],
  ["first", /first|investigate/i],
  ["top_sensor", /most|strongest|affected/i],
  ["why", /why|root.?cause|fail/i],
  ["predict", /risk|probab|predict/i],
];
export const classify = (q: string): Intent => RULES.find(([, r]) => r.test(q))?.[0] ?? "unknown";

const cache = new Map<number, AnalyzeFull>();
async function getAnalysis(id: number) { const hit = cache.get(id); if (hit) return hit; const a = await analyzeFull(id, 5); cache.set(id, a); return a; }
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const needSample = (): AssistantReply => ({ kind: "explainer", sources: [], text: "Set a sample ID in the context bar first, so I can query the backend for that wafer." });
const cand = (c: Candidate) => `${c.feature} (candidate score ${pct(c.root_cause_score)}, confidence ${c.confidence}, SHAP ${c.shap_contribution.toFixed(3)}, ${c.deviation_sigma.toFixed(2)}σ from the PASS mean)`;
const summary = (a: AnalyzeFull) => `Sample ${a.sample.sample_id} is predicted ${a.prediction.predicted_class} with ${pct(a.prediction.failure_probability)} failure risk (${a.prediction.risk_level}; decision threshold ${pct(a.prediction.decision_threshold)}).`;
const note = (a: AnalyzeFull) => [a.causality_note, a.disclaimer].filter(Boolean).join(" ");
const nonEmpty = (v: unknown) => Array.isArray(v) ? v.length > 0 : !!v && typeof v === "object" && Object.keys(v as object).length > 0;
const pick = (o: any, keys: string[]) => { for (const k of keys) if (typeof o?.[k] === "number") return o[k] as number; };

export async function getEngineerAssistantResponse(question: string, ctx: WaferContext): Promise<AssistantReply> {
  const intent = classify(question);
  const live = (text: string, sources: string[], details?: AssistantReply["details"]): AssistantReply => ({ kind: "live", text, sources, details });
  const SRC = "POST /analyze";

  // ---- explainers that need no wafer ----
  if (intent === "shap") return { kind: "explainer", sources: [], text: SHAP_TEXT };
  if (intent === "risk_meaning") return { kind: "explainer", sources: [], text: RISK_TEXT };
  if (intent === "red") return { kind: "explainer", sources: [], text: RED_TEXT };
  if (intent === "graph") {
    const fi = await getFeatureImportance(20);
    const top = fi.features.slice(0, 3).map((f) => `${f.feature} (${f.mean_abs_shap.toFixed(3)})`).join(", ");
    return { kind: "live", sources: ["GET /feature-importance"], text: `${GRAPH_TEXT} In the current model the highest-importance features are ${top}.`, details: { title: "Backend: feature importance", data: fi } };
  }
  if (intent === "model") {
    const m = await getModelInfo();
    return live(`Model: ${m.model ?? "N/A"}, version ${m.model_version ?? "N/A"}, trained ${m.training_date ?? "N/A"}, ${m.n_model_features ?? "N/A"} features, decision threshold ${m.threshold ?? "N/A"}. Each prediction is the model’s probability of FAIL for the sample’s sensor values; SHAP then splits that into per-feature contributions. Metrics are in the details below.`, ["GET /model-info"], { title: "Backend: model info", data: m });
  }
  if (intent === "defect") return { kind: "explainer", sources: ["/image/analyze", "/analyze-with-image"], text: "Live wafer-image intelligence is available on the Images page. Upload a wafer map there and the backend will return the defect class, top-3 probabilities, image anomaly signals, and linked process evidence." };
  if (intent === "unknown") return { kind: "explainer", sources: [], text: "I can answer questions about this wafer’s failure risk, root-cause candidates, SHAP contributions, similar historical samples and what-if changes. Try one of the quick questions on the right." };

  // ---- everything below needs a wafer ----
  if (ctx.sampleId === null) return needSample();
  if (intent === "predict") {
    const r = await predict(ctx.sampleId);
    return live(summary({ ...(r as any) }), ["POST /predict"]);
  }
  const a = await getAnalysis(ctx.sampleId);
  const top = a.root_cause_candidates[0];
  if (!top && intent !== "history") return live(`${summary(a)} The backend returned no root-cause candidates for this sample.`, [SRC]);

  switch (intent) {
    case "why": return live(`${summary(a)} The highest-ranked root-cause candidate is ${cand(top)}. This is a model-derived hypothesis, not a proven cause. ${top.historical_evidence ? `In ${top.historical_evidence.abnormal_same_direction} of ${top.historical_evidence.similar_fail_samples_checked} similar failed samples it was abnormal in the same direction.` : ""} ${nonEmpty(a.recommendation) ? "The backend also returned a recommendation (see details)." : ""} ${note(a)}`, [SRC], nonEmpty(a.recommendation) ? { title: "Backend: recommendation", data: a.recommendation } : undefined);
    case "top_sensor": {
      const p = a.explanation.top_positive[0];
      return live(p ? `${p.feature} has the largest push toward FAIL: SHAP ${p.contribution.toFixed(3)} (value ${p.value}). ${a.explanation.units}.` : "No positive contributions were returned.", [SRC], { title: "Backend: SHAP explanation", data: a.explanation });
    }
    case "deviation": case "first": {
      const s = top.normal_stats; const f = top.failure_stats;
      return live(`${top.feature}: current value ${top.current_value}, ${top.deviation_sigma.toFixed(2)}σ from the PASS mean${s ? ` (PASS mean ${s.mean.toFixed(2)}, 5th–95th percentile ${s.p5.toFixed(2)}–${s.p95.toFixed(2)})` : ""}${f ? `; FAIL mean ${f.mean.toFixed(2)}` : ""}.${intent === "first" ? ` Start with ${top.feature}, then the next candidates: ${a.root_cause_candidates.slice(1, 3).map((c) => c.feature).join(", ") || "none returned"}.` : ""} ${top.explanation}`, [SRC], { title: "Backend: top candidate", data: top });
    }
    case "history": {
      const h = a.historical_matches ?? [];
      const ev = top?.historical_evidence;
      return live(`${h.length ? `The backend returned ${h.length} historical match(es).` : "The backend returned no historical matches."}${ev ? ` For ${top.feature}, ${ev.abnormal_same_direction} of ${ev.similar_fail_samples_checked} similar failed samples were abnormal in the same direction.` : ""}`, [SRC], h.length ? { title: "Backend: historical matches", data: h } : undefined);
    }
    case "whatif": {
      const feature = question.match(/sensor\s*\d+/i)?.[0].replace(/sensor\s*/i, "Sensor ") ?? top.feature;
      const v = question.match(/(?:to|toward|→)\s*(-?\d+(?:\.\d+)?)/i)?.[1];
      if (v === undefined) {
        const cf = a.counterfactuals ?? [];
        return live(cf.length ? `The backend returned ${cf.length} counterfactual(s) for sample ${a.sample.sample_id} (see details). To test a specific value, ask e.g. “What if ${feature} is set to ${top.current_value}?”` : `No counterfactuals were returned. Ask with a target value, e.g. “What if ${feature} is set to <value>?”`, [SRC], cf.length ? { title: "Backend: counterfactuals", data: cf } : undefined);
      }
      if (a.sample.sample_id == null) return live("This analysis has no linked sample ID for a what-if scenario.", [SRC]);
      const r = await whatIf(a.sample.sample_id, feature, Number(v));
      const after = pick(r, ["scenario_risk", "new_probability", "scenario_probability", "counterfactual_probability", "failure_probability"]);
      return live(`Scenario: ${feature} → ${v}. ${after !== undefined ? `Predicted failure risk: ${pct(a.prediction.failure_probability)} → ${pct(after)}.` : "See the backend response below for the scenario result."} This is a model-based counterfactual and requires engineering validation before any physical process change.`, ["POST /analyze", "POST /what-if"], { title: "Backend: /what-if response", data: r });
    }
  }
  return live(summary(a), [SRC]);
}
