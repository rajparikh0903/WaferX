"""CLI:  python run_pipeline.py {train|analyze|demo|serve|synthetic|plots} ..."""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    a = sub.add_parser("analyze")
    a.add_argument("--sample-id", type=int, required=True)
    a.add_argument("--anomaly-score", type=float)
    a.add_argument("--anomaly-label", choices=["true", "false"])
    a.add_argument("--out")
    sub.add_parser("demo")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    sub.add_parser("synthetic")
    sub.add_parser("plots")
    args = ap.parse_args()

    if args.cmd == "train":
        from src.training.train_xgboost import run_training
        from src.utils import load_config
        run_training(load_config())
    elif args.cmd == "analyze":
        from src.inference.pipeline import analyze_sample
        lab = None if args.anomaly_label is None else args.anomaly_label == "true"
        out = analyze_sample(args.sample_id, args.anomaly_score, lab)
        txt = json.dumps(out, indent=2)
        if args.out:
            open(args.out, "w").write(txt)
        else:
            print(txt)
    elif args.cmd == "demo":
        from scripts.demo import run_demo
        run_demo()
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("src.inference.api:app", host="0.0.0.0", port=args.port)
    elif args.cmd == "synthetic":
        from scripts.synthetic_validation import run_synthetic
        run_synthetic()
    elif args.cmd == "plots":
        from scripts.generate_plots import generate_all
        generate_all()


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
