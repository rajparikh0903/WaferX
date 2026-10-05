"""Train the same pipeline with a stratified (random) split in a temp dir and compare against the temporal split."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.training.train_xgboost import run_training
from src.utils import load_config, resolve


def run():
    cfg = load_config()
    main_meta = json.loads((resolve(cfg, "metadata_dir") / "metadata.json").read_text())
    c2 = load_config()
    c2["split"]["method"] = "stratified"
    with tempfile.TemporaryDirectory() as tmp:
        c2["paths"] = {k: str(Path(tmp) / k) for k in c2["paths"]}
        c2["paths"]["raw_dir"] = str(Path(cfg["_root"]) / cfg["paths"]["raw_dir"])
        alt = run_training(c2)
    f = lambda m: pd.DataFrame(m["comparison"])[["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]].round(3).to_markdown(index=False)
    txt = ["# Split sensitivity (REAL SECOM)", "",
           "SECOM's fail rate drifts over time, so a random split is optimistic. Same pipeline, two splits (test metrics):", "",
           "## Temporal split (default, honest)", "", f(main_meta),
           f"\nTest FAIL prevalence: {main_meta['split']['test']['n_fail'] / main_meta['split']['test']['n']:.3f}", "",
           "## Stratified random split (optimistic)", "", f(alt),
           f"\nTest FAIL prevalence: {alt['split']['test']['n_fail'] / alt['split']['test']['n']:.3f}"]
    (resolve(cfg, "reports_dir") / "split_sensitivity.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    run()
