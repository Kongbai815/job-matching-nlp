import argparse
import json
from pathlib import Path

import pandas as pd

from .core import JobMatchingSystem


def process_batch_frame(frame, system, text_column="text", mode_column="mode"):
    if text_column not in frame.columns:
        raise ValueError(f"Input CSV must contain a '{text_column}' column.")
    rows = []
    for row_number, (_, source) in enumerate(frame.iterrows(), start=1):
        text = str(source[text_column])
        mode = str(source.get(mode_column, "auto") or "auto").lower()
        result = system.run(text, input_mode=mode)
        top = result["retrieved_evidence"][0] if result["retrieved_evidence"] else {}
        rows.append(
            {
                "input_row": row_number,
                "input_mode": result["input_mode"],
                "predicted_posting_label": result["predicted_posting_label"],
                "prediction_margin": result["prediction_margin"],
                "review_margin_threshold": result["review_margin_threshold"],
                "review_required": result["review_required"],
                "recommended_action": result["recommended_action"],
                "policy_reasons": " | ".join(result["policy_reasons"]),
                "authorization_evidence": result["input_authorization_evidence"],
                "top_posting_id": top.get("posting_id"),
                "top_role_title": top.get("role_title"),
                "top_company": top.get("company"),
                "top_country": top.get("job_country"),
                "applied_search_constraints": json.dumps(result["applied_search_constraints"]),
                "result_json": json.dumps(result),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run the MSBA job matcher over a CSV review queue.")
    parser.add_argument("--input", required=True, help="CSV with text and optional mode columns.")
    parser.add_argument("--output", default="outputs/batch.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--mode-column", default="mode")
    parser.add_argument("--data", default="jobs.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--review-policy", default="models/policy.json")
    args = parser.parse_args()

    source = pd.read_csv(args.input, keep_default_na=False)
    system = JobMatchingSystem(
        data_path=args.data,
        model_path=args.model,
        review_policy_path=args.review_policy,
    )
    completed = process_batch_frame(source, system, args.text_column, args.mode_column)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed.to_csv(output, index=False)
    print(json.dumps({"input_rows": len(source), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
