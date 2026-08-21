import argparse
import json
from pathlib import Path

import pandas as pd

from .core import JobMatchingSystem, compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MSBA job matching prototype.")
    parser.add_argument("--data", default="jobs.csv")
    parser.add_argument("--sample-per-label", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2411)
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--output", default="outputs/eval.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists() and data_path.name == "jobs.csv":
        data_path = Path("jobs.csv")

    df = pd.read_csv(data_path)
    eval_df = (
        df[df["split"].eq("validation")]
        .groupby("relevance_label", group_keys=False)
        .sample(n=args.sample_per_label, random_state=args.seed)
    )
    system = JobMatchingSystem(data_path=data_path, model_path=args.model)
    y_true = eval_df["relevance_label"].tolist()
    y_pred = [system.predict_label(text)[0] for text in eval_df["posting_text"]]
    metrics = compute_metrics(y_true, y_pred)
    result = {
        "sample_rows": len(eval_df),
        "sample_per_label": args.sample_per_label,
        "classifier_name": system.classifier_name,
        "metrics": metrics,
        "note": "Labels are weak labels derived from transparent advisor-screening rules.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
