# Solution Overview

## What We Built

**YieldTwin** is an AI-powered semiconductor manufacturing decision-support platform. It helps engineers understand why wafer yield is at risk instead of only predicting whether a wafer will pass or fail.

The system combines process and sensor analysis with wafer-image intelligence. It detects abnormal behaviour, identifies likely root causes, compares historical failures, evaluates what-if changes, and provides engineering recommendations through a unified dashboard and AI Engineer Assistant.

## How It Works

1. The engineer selects a manufacturing sample or provides process/sensor data.
2. YieldTwin predicts PASS/FAIL risk and calculates the failure probability.
3. Anomaly detection identifies unusual process behaviour.
4. Explainable analysis ranks likely root-cause sensors using SHAP, statistical deviation, and historical evidence.
5. The engineer can run a what-if scenario to see how changing a sensor value could affect failure risk.
6. For wafer images, the system generates DINOv2/ViT embeddings, classifies wafer patterns, and detects image anomalies.
7. The AI Engineer Assistant presents the model results in natural language and supports investigation.
8. IBM Bob can access the same live YieldTwin capabilities through an MCP integration.

## Architecture Diagram

> See [`architecture.md`](architecture.md) for the detailed architecture.

```text
                         ┌─────────────────┐
                         │ Engineer / User │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              Next.js Frontend              IBM Bob
                    │                           │
                    │ REST API                  │ MCP
                    ▼                           ▼
              ┌────────────────────────────────────┐
              │          FastAPI Backend            │
              └────────────────┬───────────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        Process AI        Root Cause         What-If
        + Anomaly         + SHAP             Analysis
             │
             ▼
       Wafer Image AI
       DINOv2 + ViT
       WM / MIX Classifiers
       Image Anomaly Detection
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Separate process and image intelligence** | Sensor/process data and wafer images provide different types of evidence, so each is handled by a specialized model pipeline. |
| **Explainable root-cause analysis** | Engineers need evidence behind a prediction, not only a PASS/FAIL result. SHAP, statistical deviation, and historical similarity are used to rank candidate causes. |
| **Counterfactual what-if analysis** | Engineers can evaluate possible parameter changes using model-based scenarios before considering a real process change. |
| **FastAPI + Next.js architecture** | Keeps ML inference and API logic separate from the user interface and makes the services easy to test and integrate. |
| **IBM Bob through MCP** | Allows IBM Bob to invoke the real YieldTwin backend tools instead of being only a development assistant. |

## IBM Technologies Used

- **IBM Bob:** Used as a load-bearing engineering agent through a project-level MCP integration. Bob can invoke YieldTwin tools for live sample analysis, anomaly detection, root-cause analysis, feature importance, and counterfactual/what-if analysis.
- **MCP Integration:** A YieldTwin MCP server bridges IBM Bob to the FastAPI backend, allowing Bob to use the same live ML services as the web application.

> **Note:** The current project does **not** use watsonx.ai, IBM Cloud, IBM databases, or Slack as runtime dependencies. IBM Bob + MCP is the IBM technology integration implemented in the current prototype.
