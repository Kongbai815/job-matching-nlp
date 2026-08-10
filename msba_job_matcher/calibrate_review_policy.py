import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prediction_margins(model, texts):
    decisions = np.asarray(model.decision_function(texts), dtype=float)
    ordered = np.sort(decisions, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def largest_coverage_threshold(y_true, predicted, margins, label, target_precision):
    selected = np.flatnonzero(predicted == label)
    label_margins = margins[selected]
    correct = (y_true[selected] == predicted[selected]).astype(int)
    order = np.argsort(-label_margins)
    sorted_margins = label_margins[order]
    cumulative_correct = np.cumsum(correct[order])
    tie_ends = np.r_[np.flatnonzero(sorted_margins[:-1] != sorted_margins[1:]), len(order) - 1]
    candidates = []
    for end in tie_ends:
        accepted = int(end + 1)
        precision = float(cumulative_correct[end] / accepted)
        if precision >= target_precision:
            candidates.append((accepted, float(sorted_margins[end]), precision))
    if not candidates:
        raise ValueError(f"No threshold reaches target precision for {label}.")
    accepted, threshold, precision = max(candidates, key=lambda item: item[0])
    return {
        "predicted_rows": int(len(selected)),
        "accepted_rows": accepted,
        "coverage": accepted / len(selected),
        "precision": precision,
        "margin_threshold": threshold,
    }


def evaluate_policy(y_true, predicted, margins, thresholds):
    accepted = np.array(
        [margin >= thresholds[str(label)] for label, margin in zip(predicted, margins)],
        dtype=bool,
    )
    per_label = {}
    for label in sorted(thresholds):
        selected = predicted == label
        kept = selected & accepted
        per_label[label] = {
            "predicted_rows": int(selected.sum()),
            "accepted_rows": int(kept.sum()),
            "coverage": float(kept.sum() / selected.sum()) if selected.any() else 0.0,
            "precision": float((y_true[kept] == predicted[kept]).mean()) if kept.any() else 0.0,
        }
    return {
        "rows": int(len(y_true)),
        "accepted_rows": int(accepted.sum()),
        "mandatory_review_rows": int((~accepted).sum()),
        "coverage": float(accepted.mean()),
        "mandatory_review_rate": float((~accepted).mean()),
        "selective_accuracy": float((y_true[accepted] == predicted[accepted]).mean()),
        "per_predicted_label": per_label,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate class-specific mandatory-review thresholds.")
    parser.add_argument("--data", default="data_jobs_msba_project_sample_100k.csv")
    parser.add_argument("--model", default="models/job_fit_tfidf_svc.joblib")
    parser.add_argument("--output", default="models/review_policy.json")
    parser.add_argument("--target-precision", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=2411)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    validation = frame[frame["split"].eq("validation")].reset_index(drop=True)
    model = joblib.load(args.model)
    texts = validation["posting_text"].fillna("")
    y_true = validation["relevance_label"].astype(str).to_numpy()
    predicted = np.asarray(model.predict(texts), dtype=str)
    margins = prediction_margins(model, texts)
    calibration_idx, audit_idx = train_test_split(
        np.arange(len(validation)),
        test_size=0.5,
        random_state=args.seed,
        stratify=y_true,
    )

    calibration = {}
    thresholds = {}
    for label in [str(value) for value in model.classes_]:
        block = largest_coverage_threshold(
            y_true[calibration_idx],
            predicted[calibration_idx],
            margins[calibration_idx],
            label,
            args.target_precision,
        )
        calibration[label] = block
        thresholds[label] = block["margin_threshold"]

    payload = {
        "policy_name": "class-specific LinearSVC margin policy",
        "metric_scope": "agreement with project weak labels",
        "target_calibration_precision": args.target_precision,
        "seed": args.seed,
        "calibration_rows": int(len(calibration_idx)),
        "audit_rows": int(len(audit_idx)),
        "per_label_margin_thresholds": thresholds,
        "calibration": calibration,
        "independent_audit": evaluate_policy(
            y_true[audit_idx],
            predicted[audit_idx],
            margins[audit_idx],
            thresholds,
        ),
        "policy_boundary": (
            "A margin below the predicted class threshold forces mandatory advisor review. "
            "Authorization restrictions and missing fields remain separate deterministic gates."
        ),
        "limitation": "Thresholds are calibrated against weak labels, not advisor ground truth.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
