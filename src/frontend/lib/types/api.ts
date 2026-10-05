export type Stats = { n: number; mean: number; median: number; std: number; p5: number; p95: number };
export interface Sample { sample_id: number | null; split: string; timestamp?: string; actual_label?: string; n_model_features: number; note?: string | null }
export interface Prediction { sample_id: number | null; predicted_class: string; failure_probability: number; risk_level: string; decision_threshold: number }
export interface Contribution { feature: string; value: number; contribution: number; direction: string }
export interface Explanation { units: string; base_value: number; logit: number; failure_probability: number; top_positive: Contribution[]; top_negative: Contribution[] }
export interface Candidate {
  feature: string; current_value: number; normal_stats?: Stats; failure_stats?: Stats;
  shap_contribution: number; shap_rank: number; deviation_sigma: number; deviation_percentile_in_pass?: number;
  failure_association?: { effect_size_r: number; auc: number; p_value: number; q_value: number; significant: boolean };
  historical_evidence?: { similar_fail_samples_checked: number; abnormal_same_direction: number };
  root_cause_score: number; confidence: string; evidence_strength: string; label: string; explanation: string;
}
export interface AnomalyInfo {
  available: boolean; sample_id?: number | null; anomaly_score?: number | null; anomaly_label?: boolean | null;
  effective_score?: number | null; source?: string | null; model?: string; model_type?: string;
  alert_threshold?: number; severity?: string; score_semantics?: string; train_pass_count?: number;
  train_split_size?: number; evaluation_note?: string;
}
export interface AnalyzeResponse {
  sample: Sample; prediction: Prediction; anomaly?: AnomalyInfo; explanation: Explanation; root_cause_candidates: Candidate[];
  historical_matches?: unknown[]; causality_note?: string; disclaimer?: string;
  counterfactuals?: Loose[]; multi_parameter_experiments?: Loose[]; recommendation?: Loose;
}
export interface FeatureImportance { units: string; features: { rank: number; feature: string; mean_abs_shap: number; mean_shap: number; share: number }[] }
export type ModelInfo = Record<string, any>;
export interface HealthResponse { status: string; root_cause: string; anomaly_detection: string; image_intelligence: string; image_datasets: string[] }
export interface ImageModelInfo {
  dataset: string; classes: string[]; normal_class: string; backbones: Record<string, string>;
  embedding_tags: string[]; in_dim: number; backbone_weights: string;
}
export interface ImageAnalysisResult {
  dataset: string; label: string; confidence: number; top3: Record<string, number>; normal_class: string;
  normal_probability: number; anomaly_score_classifier: number; anomaly_label: boolean; detector_score: number;
  anomaly_score_semantics: string; detector_score_semantics: string;
  input?: { filename?: string; map_shape?: number[] };
}
export interface CombinedAnalysis {
  process_intelligence: AnalyzeFull;
  wafer_image_intelligence: ImageAnalysisResult;
  integration: { process_model: string; image_model: string };
}
export type Loose = Record<string, any>;
export interface AnalyzeFull extends AnalyzeResponse { counterfactuals?: Loose[]; multi_parameter_experiments?: Loose[]; recommendation?: Loose }
