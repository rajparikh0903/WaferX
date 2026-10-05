"""Smoke-check the exported image model stack before a hackathon demo."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "image_intelligence"

FILES = [
    "classifier_mlp_wm_full.pt",
    "classifier_mlp_wm_full_prep.joblib",
    "detector_wm_full_dino+vit.joblib",
    "classifier_mlp_mix_full.pt",
    "classifier_mlp_mix_full_prep.joblib",
    "detector_mix_full_dino+vit.joblib",
]

print("YieldTwin image-intelligence smoke check")
print("Model directory:", MODEL_DIR)
missing = [name for name in FILES if not (MODEL_DIR / name).exists()]
print("Artifacts:", "OK" if not missing else f"MISSING {missing}")
for pkg in ("torch", "timm", "cv2"):
    print(f"{pkg}:", "installed" if importlib.util.find_spec(pkg) else "MISSING")

if missing:
    raise SystemExit(1)
if not importlib.util.find_spec("timm"):
    print("NOTE: install the image requirements before inference.")
    raise SystemExit(2)

from src.image_intelligence import image_model_service  # noqa: E402

for dataset in ("wm", "mix"):
    info = image_model_service.info(dataset)
    print(dataset, "classes:", len(info["classes"]), "device:", info["device"])

print("Image model configuration is ready. The first inference may download/cache timm backbone weights.")
