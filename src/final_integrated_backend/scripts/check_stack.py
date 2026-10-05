"""Quick offline health check for the final YieldTwin inference stack."""
from __future__ import annotations

import json
from src.inference.pipeline import RootCausePipeline


def main() -> None:
    pipe = RootCausePipeline()
    print("ROOT CAUSE: READY")
    print("ANOMALY: READY")
    print("ANOMALY INFO:")
    print(json.dumps(pipe.anomaly_detector.info(), indent=2))

    for sid in (11, 23, 500):
        out = pipe.analyze(sample_id=sid, top_k=5)
        print(f"\nSAMPLE {sid}")
        print(json.dumps({
            "prediction": out["prediction"],
            "anomaly": out["anomaly"],
            "top_root_cause": out["root_cause_candidates"][0],
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
