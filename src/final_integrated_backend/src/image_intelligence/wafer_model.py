"""Wafer-map image classification and visual anomaly inference.

This module wraps the exported WM/MIX classifier artifacts from wafer_export
without changing the trained model itself. The original model uses DINOv2 and
ViT embeddings followed by a trained MLP classifier plus a PCA + Isolation
Forest image detector.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Literal

import cv2
import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import timm
except Exception as exc:  # pragma: no cover - environment dependent
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


BACKBONES = {
    "dino": ("vit_base_patch14_dinov2.lvd142m", {"img_size": 224}),
    "vit": ("vit_base_patch16_224.augreg_in21k_ft_in1k", {}),
}

DatasetName = Literal["wm", "mix"]


def _require_image_dependencies() -> None:
    if torch is None or timm is None:
        raise RuntimeError(
            "Image intelligence dependencies are unavailable. Install the image "
            "requirements (torch, timm, opencv-python-headless)."
        ) from (_TORCH_IMPORT_ERROR if torch is None else _TIMM_IMPORT_ERROR)


def _to_training_map(arr: np.ndarray) -> np.ndarray:
    """Convert input to the original 0/1/2 wafer-map representation."""
    a = np.asarray(arr)

    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            a = cv2.cvtColor(
                a[..., :3].astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )
        else:
            a = np.squeeze(a)

    if a.ndim != 2:
        raise ValueError(f"Wafer map must be 2-D; received shape {a.shape}")

    if not np.isfinite(a.astype(np.float32)).all():
        raise ValueError("Wafer map contains non-finite values")

    # Preserve the exact representation expected by the original model.
    unique = np.unique(a)

    if np.all(np.isin(unique, [0, 1, 2])):
        return a.astype(np.uint8)

    # Original wafer_infer.py visualizes map values with value * 127,
    # producing 0, 127 and 254. Recover those discrete values instead
    # of collapsing the image to binary at a 127 threshold.
    a = np.asarray(a, dtype=np.float32)

    return np.clip(
        np.rint(a / 127.0),
        0,
        2,
    ).astype(np.uint8)


def decode_wafer_bytes(
    data: bytes,
    filename: str | None = None,
) -> np.ndarray:
    """Decode .npy/.npz or a normal image upload into a 2-D wafer map."""

    suffix = Path(filename or "").suffix.lower()

    if suffix == ".npy":
        return _to_training_map(
            np.load(
                io.BytesIO(data),
                allow_pickle=False,
            )
        )

    if suffix == ".npz":
        with np.load(
            io.BytesIO(data),
            allow_pickle=False,
        ) as z:
            if not z.files:
                raise ValueError("NPZ file does not contain an array")

            return _to_training_map(z[z.files[0]])

    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError(
            "Could not decode the wafer map. Upload PNG/JPG/WebP, NPY, or NPZ."
        )

    if image.ndim == 3 and image.shape[2] >= 3:
        image = cv2.cvtColor(
            image[..., :3],
            cv2.COLOR_BGR2GRAY,
        )

    return _to_training_map(image)


class WaferImageModel:
    """Lazy-loading wrapper around one exported WM or MIX model family."""

    def __init__(
        self,
        folder: str | Path,
        ds: DatasetName,
        device: str | None = None,
    ):
        _require_image_dependencies()

        self.folder = Path(folder)
        self.ds = ds
        self.dev = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dt = (
            torch.float16
            if self.dev == "cuda"
            else torch.float32
        )

        ck_path = self.folder / f"classifier_mlp_{ds}_full.pt"
        pp_path = self.folder / f"classifier_mlp_{ds}_full_prep.joblib"
        det_path = self.folder / f"detector_{ds}_full_dino+vit.joblib"

        for p in (ck_path, pp_path, det_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Image model artifact not found: {p}"
                )

        ck = torch.load(
            ck_path,
            map_location="cpu",
            weights_only=False,
        )

        pp = joblib.load(pp_path)
        self.det = joblib.load(det_path)

        self.classes = list(ck["classes"])
        self.tags = list(pp["tags"])
        self.scaler = pp["scaler"]

        self.normal = (
            "none"
            if "none" in self.classes
            else "00000000"
        )

        self.net = nn.Sequential(
            nn.Linear(ck["in_dim"], 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.classes)),
        )

        self.net.load_state_dict(ck["state"])
        self.net.eval()

        self.models: dict[str, Any] = {}

        for tag in sorted(
            set(self.tags) | set(self.det["tags"])
        ):
            name, kwargs = BACKBONES[tag]

            # The exported classifier contains the trained MLP heads,
            # while the DINO/ViT backbones are the pretrained timm
            # models used at training.
            self.models[tag] = (
                timm.create_model(
                    name,
                    pretrained=True,
                    num_classes=0,
                    **kwargs,
                )
                .to(self.dev)
                .eval()
                .to(self.dt)
            )

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.dev,
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.dev,
        ).view(1, 3, 1, 1)

    @staticmethod
    def _to_img(m: np.ndarray) -> np.ndarray:
        return cv2.resize(
            np.asarray(m).astype(np.uint8) * 127,
            (224, 224),
            interpolation=cv2.INTER_NEAREST,
        )

    def embed(
        self,
        maps: list[np.ndarray],
        batch_size: int = 64,
    ) -> dict[str, np.ndarray]:
        out = {tag: [] for tag in self.models}

        with torch.no_grad():
            for start in range(0, len(maps), batch_size):
                batch = maps[start : start + batch_size]

                x = torch.from_numpy(
                    np.stack(
                        [
                            self._to_img(m)
                            for m in batch
                        ]
                    )
                ).to(self.dev).float() / 254.0

                x = (
                    (
                        x[:, None]
                        .repeat(1, 3, 1, 1)
                        - self.mean
                    )
                    / self.std
                ).to(self.dt)

                for tag, model in self.models.items():
                    out[tag].append(
                        model(x)
                        .float()
                        .cpu()
                        .numpy()
                    )

        return {
            tag: np.concatenate(values)
            for tag, values in out.items()
        }

    def predict(
        self,
        maps: np.ndarray | list[np.ndarray],
    ) -> list[dict[str, Any]]:
        if isinstance(maps, np.ndarray) and maps.ndim == 2:
            maps = [maps]

        maps = [
            _to_training_map(m)
            for m in maps
        ]

        embeddings = self.embed(list(maps))

        X = self.scaler.transform(
            np.hstack(
                [
                    embeddings[tag]
                    for tag in self.tags
                ]
            )
        ).astype(np.float32)

        with torch.no_grad():
            probs = torch.softmax(
                self.net(torch.tensor(X)),
                dim=1,
            ).cpu().numpy()

        detector = self.det

        Z = np.hstack(
            [
                scaler.transform(
                    embeddings[tag]
                )
                / np.sqrt(
                    embeddings[tag].shape[1]
                )
                for tag, scaler in zip(
                    detector["tags"],
                    detector["scalers"],
                )
            ]
        )

        detector_scores = -detector["iso"].score_samples(
            detector["pca"].transform(Z)
        )

        normal_index = self.classes.index(self.normal)

        results: list[dict[str, Any]] = []

        for p, detector_score in zip(
            probs,
            detector_scores,
        ):
            top = np.argsort(-p)[:3]

            anomaly_score = float(
                1.0 - p[normal_index]
            )

            results.append(
                {
                    "dataset": self.ds,
                    "label": self.classes[int(top[0])],
                    "confidence": round(
                        float(p[top[0]]),
                        4,
                    ),
                    "top3": {
                        self.classes[int(j)]: round(
                            float(p[int(j)]),
                            4,
                        )
                        for j in top
                    },
                    "normal_class": self.normal,
                    "normal_probability": round(
                        float(p[normal_index]),
                        4,
                    ),
                    "anomaly_score_classifier": round(
                        anomaly_score,
                        4,
                    ),
                    "anomaly_label": bool(
                        anomaly_score >= 0.5
                    ),
                    "detector_score": round(
                        float(detector_score),
                        4,
                    ),
                    "anomaly_score_semantics": (
                        "1 - classifier probability "
                        "of normal class"
                    ),
                    "detector_score_semantics": (
                        "raw image Isolation Forest score; "
                        "higher is more unusual"
                    ),
                }
            )

        return results

    def predict_bytes(
        self,
        data: bytes,
        filename: str | None = None,
    ) -> dict[str, Any]:
        wafer_map = decode_wafer_bytes(
            data,
            filename,
        )

        result = self.predict(
            wafer_map
        )[0]

        result["input"] = {
            "filename": filename,
            "map_shape": [
                int(wafer_map.shape[0]),
                int(wafer_map.shape[1]),
            ],
        }

        return result

    def info(self) -> dict[str, Any]:
        return {
            "dataset": self.ds,
            "classes": self.classes,
            "normal_class": self.normal,
            "backbones": {
                tag: BACKBONES[tag][0]
                for tag in sorted(
                    set(self.tags)
                    | set(self.det["tags"])
                )
            },
            "device": self.dev,
            "embedding_tags": self.tags,
        }


class ImageModelService:
    """Lazily creates WM/MIX model instances so the normal API remains lightweight."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self._models: dict[str, WaferImageModel] = {}

    def get(
        self,
        dataset: DatasetName = "wm",
    ) -> WaferImageModel:
        if dataset not in ("wm", "mix"):
            raise ValueError(
                "dataset must be 'wm' or 'mix'"
            )

        if dataset not in self._models:
            self._models[dataset] = WaferImageModel(
                self.folder,
                dataset,
            )

        return self._models[dataset]

    def predict_bytes(
        self,
        data: bytes,
        filename: str | None = None,
        dataset: DatasetName = "wm",
    ) -> dict[str, Any]:
        return self.get(dataset).predict_bytes(
            data,
            filename,
        )

    def info(
        self,
        dataset: DatasetName = "wm",
    ) -> dict[str, Any]:
        # Read lightweight metadata without instantiating
        # DINO/ViT backbones.
        if dataset not in ("wm", "mix"):
            raise ValueError(
                "dataset must be 'wm' or 'mix'"
            )

        if torch is None:
            raise RuntimeError(
                "Image intelligence dependencies are unavailable. "
                "Install the image requirements "
                "(torch, timm, opencv-python-headless)."
            ) from _TORCH_IMPORT_ERROR

        ck_path = (
            self.folder
            / f"classifier_mlp_{dataset}_full.pt"
        )

        pp_path = (
            self.folder
            / f"classifier_mlp_{dataset}_full_prep.joblib"
        )

        det_path = (
            self.folder
            / f"detector_{dataset}_full_dino+vit.joblib"
        )

        for p in (
            ck_path,
            pp_path,
            det_path,
        ):
            if not p.exists():
                raise FileNotFoundError(
                    f"Image model artifact not found: {p}"
                )

        ck = torch.load(
            ck_path,
            map_location="cpu",
            weights_only=False,
        )

        pp = joblib.load(pp_path)
        det = joblib.load(det_path)

        classes = list(ck["classes"])

        normal = (
            "none"
            if "none" in classes
            else "00000000"
        )

        return {
            "dataset": dataset,
            "classes": classes,
            "normal_class": normal,
            "backbones": {
                tag: BACKBONES[tag][0]
                for tag in sorted(
                    set(pp["tags"])
                    | set(det["tags"])
                )
            },
            "embedding_tags": list(
                pp["tags"]
            ),
            "in_dim": int(
                ck["in_dim"]
            ),
            "backbone_weights": (
                "pretrained timm weights "
                "required at inference time"
            ),
        }


MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "image_intelligence"
)

image_model_service = ImageModelService(
    MODEL_DIR
)