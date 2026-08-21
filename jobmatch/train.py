import argparse
import json
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


LABELS = ["high_fit", "medium_fit", "low_fit", "unclear"]


def normalize_text(value):
    return re.sub(r"\W+", " ", str(value).lower()).strip()


def make_pipeline(random_state=2411):
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=3,
                                max_df=0.98,
                                max_features=70_000,
                                sublinear_tf=True,
                                strip_accents="unicode",
                                stop_words="english",
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=3,
                                max_features=70_000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LinearSVC(C=1.0, random_state=random_state)),
        ]
    )


def metric_block(frame, predictions):
    report = classification_report(
        frame["relevance_label"], predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "rows": len(frame),
        "accuracy": accuracy_score(frame["relevance_label"], predictions),
        "macro_f1": f1_score(frame["relevance_label"], predictions, average="macro"),
        "per_label_f1": {label: report[label]["f1-score"] for label in LABELS},
    }


def main():
    parser = argparse.ArgumentParser(description="Train the production job-fit classifier.")
    parser.add_argument("--data", default="jobs.csv")
    parser.add_argument("--model-output", default="models/model.joblib")
    parser.add_argument("--metrics-output", default="models/metrics.json")
    parser.add_argument("--seed", type=int, default=2411)
    args = parser.parse_args()

    started = time.time()
    df = pd.read_csv(args.data)
    df["text_key"] = df["posting_text"].fillna("").map(normalize_text)
    df["company_key"] = df["company"].fillna("").map(normalize_text)
    train = df[df["split"].eq("train")].drop_duplicates("text_key", keep="first").copy()
    validation = df[df["split"].eq("validation")].copy()
    fixed = validation.groupby("relevance_label", group_keys=False).sample(n=1000, random_state=args.seed)

    pipeline = make_pipeline(random_state=args.seed)
    pipeline.fit(train["posting_text"].fillna(""), train["relevance_label"])

    train_text = set(train["text_key"])
    train_company = set(train["company_key"]) - {""}
    evaluation_sets = {
        "fixed_4k": fixed,
        "fixed_4k_novel_text": fixed[~fixed["text_key"].isin(train_text)],
        "fixed_4k_unseen_company": fixed[~fixed["company_key"].isin(train_company)],
        "full_20k": validation,
        "full_20k_novel_text": validation[~validation["text_key"].isin(train_text)],
        "full_20k_unseen_company": validation[~validation["company_key"].isin(train_company)],
    }
    evaluation = {
        name: metric_block(frame, pipeline.predict(frame["posting_text"].fillna("")))
        for name, frame in evaluation_sets.items()
    }
    payload = {
        "model": "word+character TF-IDF LinearSVC",
        "train_rows_before_deduplication": int((df["split"].eq("train")).sum()),
        "train_rows_after_exact_text_deduplication": len(train),
        "fit_and_evaluation_seconds": round(time.time() - started, 1),
        "evaluation": evaluation,
        "limitations": [
            "Targets are weak labels derived from posting metadata rather than advisor judgments.",
            "Novel-text and unseen-company slices reduce but do not eliminate rule-label leakage.",
        ],
    }
    model_path = Path(args.model_output)
    metrics_path = Path(args.metrics_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
