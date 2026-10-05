export interface WaferContext { sampleId: number | null }
export interface AssistantReply {
  text: string;
  sources: string[];            // backend endpoints actually called, e.g. "POST /analyze"
  kind: "live" | "explainer";   // explainer = static help text, not model output
  details?: { title: string; data: unknown };
}
