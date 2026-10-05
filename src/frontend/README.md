# Yield Intelligence — IBM × VGEC Hackathon frontend

1. Copy `.env.example` to `.env.local` if the backend is not on `http://localhost:8000`.
2. `npm install`
3. `npm run dev`

This frontend is wired to the integrated YieldTwin FastAPI backend. Live routes include `/health`, `/model-info`, `/feature-importance`, `/predict`, `/root-cause`, `/analyze`, `/anomaly`, `/anomaly-info`, `/what-if`, `/image/model-info`, `/image/analyze`, and `/analyze-with-image`.

The investigation, what-if, image-analysis, graphs, models, and assistant flows use backend responses rather than the previous hardcoded demo model outputs.
