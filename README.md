# 🚀 Yield Intelligence (WaferX)

AI-assisted semiconductor defect investigation: from a failing sample to explainable root-cause candidates and what-if simulation.

## 👥 Team

| Field | Value |
|---|---|
| **Team Name** | WaferX |
| **Track** | AI |
| **Team Lead** | Parikh Raj M — rparikh0903@gmail.com |
| **Members** | Gundigara Shrut Pareshbhai, Patel Megha Sajeshkumar, Tailor Ishita Hemantkumar, Rana Khushi Dashrathbhai, Mewada Riya Dinesh |

---

## 🎯 Problem Statement

Semiconductor process engineers need to investigate potential causes of wafer or process failures across large numbers of equipment sensors and process parameters. Manual correlation of these signals makes defect investigation slow, and repeated defect patterns may not feed back into a systematic investigation and correction workflow. This project addresses the *Semiconductor Defect Root Cause Assistant* challenge.

---

## 💡 Solution

WaferX is an AI-assisted defect investigation workstation. It combines failure-risk prediction (LightGBM), SHAP-based explanations, ranked root-cause candidates backed by supporting evidence and historical matches, and model-based what-if simulation. Engineers can use the web dashboard or chat with IBM Bob, which is connected to the same backend through an MCP server, to move from a potentially failing sample to contributing parameters and possible parameter changes.
---

## ✨ Key Features

- **Failure-risk prediction with explanations:** LightGBM model on the UCI SECOM dataset (test ROC-AUC 0.745, PR-AUC 0.256) with a high-sensitivity screening threshold, plus SHAP local and global feature importance.
- **Ranked root-cause candidates:** combines SHAP contribution, sensor deviation, statistical association and historical similarity.
- **Historical evidence and what-if analysis:** similar past cases plus constrained counterfactual simulation of parameter changes.
- **In-app AI Engineer Assistant:** routes investigation questions to the backend capabilities.
- **IBM Bob chatbot integration:** a FastMCP server (`src/bob_mcp/`) exposes five tools (`analyze_sample`, `get_anomaly`, `get_root_causes`, `run_what_if`, `get_feature_importance`) to IBM Bob, and a custom `yield-investigation` skill guides Bob through the investigation steps.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python, TypeScript |
| **Frameworks** | FastAPI, Next.js 14, React, LightGBM, scikit-learn, FastMCP |
| **IBM Technologies** | IBM Bob (chat agent connected through a custom MCP server, skill and rules) |
| **Databases** | None (file-based: Parquet, CSV, joblib model artifacts) |
| **Other** | Model Context Protocol (MCP), SHAP, Tailwind CSS, Recharts, Lucide, SECOM dataset |
---

## 📁 Repository Structure

```
bob-ai-hackathon-WaferX/
├── .bob/                           # IBM Bob configuration
│   ├── mcp.json                    # MCP server registration (YieldTwin)
│   ├── rules-agent/
│   ├── rules-ask/
│   ├── rules-plan/
│   └── skills/
│       └── yield-investigation/    # Custom Bob investigation skill
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/                  # Submission validation workflow
├── demo/
│   ├── screenshots/                # App screenshots
│   ├── demo-video-link.txt         # Link to demo video
│   └── live-demo-url.txt           # Live demo URL
├── docs/
│   ├── problem-statement.md
│   ├── solution-overview.md
│   ├── architecture.md
│   ├── setup-guide.md
│   └── template-guide.md           # Organizer guide
├── presentation/                   # Slide deck
├── src/
│   ├── bob_mcp/                    # FastMCP server connecting IBM Bob to the backend
│   │   ├── __init__.py
│   │   ├── requirements.txt
│   │   └── server.py
│   ├── final_integrated_backend/   # FastAPI backend
│   │   ├── src/                    # API, data, root-cause, explainability, counterfactual, similarity, image modules
│   │   ├── data/                   # SECOM raw data and processed splits
│   │   ├── models/                 # Trained model artifacts
│   │   ├── benchmarks/             # Benchmark results
│   │   ├── reports/                # Validation reports
│   │   ├── scripts/                # Training and utility scripts
│   │   ├── notebooks/
│   │   ├── tests/
│   │   ├── config.yaml
│   │   ├── requirements.txt
│   │   └── start_backend.bat
│   ├── frontend/                   # Next.js dashboard
│   ├── .env.example
│   ├── README.md
│   └── wafer images to test.zip    # Sample wafer images for testing
├── .gitignore
├── AGENTS.md                       # Project guidance for IBM Bob
├── CONTRIBUTING.md
├── README.md
└── submission.yaml                 # Structured submission metadata
```

---

## ⚡ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/rajparikh0903/bob-ai-hackathon-WaferX
cd bob-ai-hackathon-WaferX

# 2. Install dependencies
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Backend (terminal 1)
cd src/final_integrated_backend
python -m pip install -r requirements.txt python -m uvicorn src.inference.api:app --reload --host 127.0.0.1 --port 8000

# 3. Frontend (terminal 2)
cd src/frontend
npm install
npm run dev

# 4. IBM Bob Integration

IBM Bob is integrated through the project-level MCP server.
From the repository root:

pip install -r src/bob_mcp/requirements.txt
```

---

## 🖥️ Demo

| Artifact | Link |
|---|---|
| 📹 Demo Video | [See demo/demo-video-link.txt](demo/demo-video-link.txt) |
| 🌐 Live Demo | NOT DEPLOYED — run locally using docs/setup-guide.md |
| 🖼️ Screenshots | [See demo/screenshots/](demo/screenshots/) |
| 📊 Presentation | [See presentation/](presentation/) |

---

## ⚠️ Known Limitations

- The anomaly detection panel and defect-image classification include placeholder/mock functionality because their real APIs are still under development.
- The sample list uses demo sample IDs from the SECOM dataset.
- Root-cause outputs are model-based hypotheses, not confirmed physical causes, and SECOM has no ground-truth causal labels.
- What-if results are model-based counterfactuals and need engineering validation before any physical process change.
- The IBM Bob MCP configuration (`.bob/mcp.json`) contains local Windows paths that must be edited on another machine.
- Not deployed: run locally using the steps above.

---

## 🏅 What We're Most Proud Of

The explainable root-cause engine. It ranks candidate sensors using several independent evidence sources and was validated on synthetic data with known injected causes (Top-1 0.953, Top-3 1.000). It is available both in a working dashboard and through an IBM Bob chatbot integration whose skill forbids inventing model outputs and requires labeling results as hypotheses that need engineering validation.

---