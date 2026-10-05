"""
Preload the pretrained timm DINOv2/ViT backbones used by the wafer models.

Run from the backend root:
    python scripts/preload_image_backbones.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_intelligence import image_model_service


def main() -> None:
    for dataset in ("wm", "mix"):
        print(f"Loading {dataset.upper()} image model...")

        model = image_model_service.get(dataset)

        print(f"{dataset.upper()} ready on {model.dev}")

    print("All wafer image backbones are initialized/cached.")


if __name__ == "__main__":
    main()