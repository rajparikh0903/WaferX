# YieldTwin Frontend ↔ Backend API Integration

The frontend targets `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

| Backend endpoint | Frontend integration |
|---|---|
| `GET /health` | Navigation status + Models health panel |
| `GET /model-info` | Models page + AI Engineer Assistant |
| `GET /feature-importance` | Graphs + Models + Assistant |
| `POST /predict` | AI Engineer Assistant |
| `POST /root-cause` | Investigate refresh button |
| `POST /analyze` | Investigate + Graphs + What-if + Assistant |
| `POST /anomaly` | Investigate live anomaly panel |
| `GET /anomaly-info` | Models page |
| `POST /what-if` | What-if page + Investigation live check + Assistant |
| `GET /image/model-info` | Images page + Models page |
| `POST /image/analyze` | Images page image-only analysis |
| `POST /analyze-with-image` | Images page combined process + image analysis |

## Run

### Backend
Start the integrated FastAPI service on port 8000.

### Frontend

```bash
copy .env.example .env.local
npm install
npm run dev
```

The UI runs on `http://localhost:3000` by default.
