import type {
  AnalyzeFull,
  AnalyzeResponse,
  AnomalyInfo,
  FeatureImportance,
  HealthResponse,
  ImageAnalysisResult,
  ImageModelInfo,
  ModelInfo,
} from "@/lib/types/api";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(public kind: "offline" | "validation" | "not_found" | "unavailable" | "server" | "empty" | "timeout", message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 120000): Promise<T> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  const headers = new Headers(init.headers);
  if (!(init.body instanceof FormData) && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: ctl.signal, headers });
  } catch (e: any) {
    if (e?.name === "AbortError") throw new ApiError("timeout", "The backend took too long to respond.");
    throw new ApiError("offline", `Unable to reach ${API_BASE_URL}. Check that FastAPI is running.`);
  } finally {
    clearTimeout(t);
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? "");
    if (res.status === 422) throw new ApiError("validation", `Invalid request. ${detail}`);
    if (res.status === 404) throw new ApiError("not_found", `Resource not found. ${detail}`);
    if (res.status === 503) throw new ApiError("unavailable", `Model unavailable. ${detail}`);
    throw new ApiError("server", `Server error (${res.status}). ${detail}`);
  }

  const data = await res.json().catch(() => null);
  if (!data || (typeof data === "object" && Object.keys(data).length === 0)) throw new ApiError("empty", "The backend returned an empty response.");
  return data as T;
}

const post = <T,>(p: string, body: unknown) => request<T>(p, { method: "POST", body: JSON.stringify(body) });

export const health = () => request<HealthResponse>("/health", {}, 100000);
export const getModelInfo = () => request<ModelInfo>("/model-info");
export const getFeatureImportance = (topN = 20) => request<FeatureImportance>(`/feature-importance?top_n=${topN}`);
export const getAnomalyInfo = () => request<Record<string, any>>("/anomaly-info");
export const getImageModelInfo = (dataset: "wm" | "mix" = "wm") => request<ImageModelInfo>(`/image/model-info?dataset=${dataset}`);

export const predict = (sample_id: number) => post<Pick<AnalyzeResponse, "sample" | "prediction">>("/predict", { sample_id });
export const rootCause = (sample_id: number, top_k = 5) => post<AnalyzeResponse>("/root-cause", { sample_id, top_k });
export const analyze = (sample_id: number, top_k = 5) => post<AnalyzeFull>("/analyze", { sample_id, top_k });
export const anomaly = (sample_id: number) => post<AnomalyInfo>("/anomaly", { sample_id });

export const whatIf = (sample_id: number, feature: string, values: number | number[], signal?: AbortSignal) => {
  const body = { sample_id, feature, values: Array.isArray(values) ? values : [values] };
  return request<Record<string, any>>("/what-if", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
};

export const analyzeFull = (sample_id: number, top_k = 5) => analyze(sample_id, top_k);

export async function imageAnalyze(file: File, dataset: "wm" | "mix" = "wm"): Promise<ImageAnalysisResult> {
  const form = new FormData();
  form.append("wafer_map", file);
  form.append("dataset", dataset);
  return request<ImageAnalysisResult>("/image/analyze", { method: "POST", body: form }, 180000);
}

export async function analyzeWithImage(file: File, sampleId: number, dataset: "wm" | "mix" = "wm", topK = 5) {
  const form = new FormData();
  form.append("wafer_map", file);
  form.append("dataset", dataset);
  form.append("sample_id", String(sampleId));
  form.append("top_k", String(topK));
  return request<import("@/lib/types/api").CombinedAnalysis>("/analyze-with-image", { method: "POST", body: form }, 180000);
}
