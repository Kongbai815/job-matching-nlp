import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .train import LABELS, make_pipeline, normalize_text


AUTH_KEYWORDS = re.compile(r"\b(?:opt|cpt|h-?1b|visa|sponsorship|work authorization|authorized to work)\b", re.I)
AUTH_RESTRICTION = re.compile(
    r"\bno\s+(?:h-?1b|opt|cpt|visa|sponsorship)|"
    r"\b(?:h-?1b|opt|cpt|visa|sponsorship).{0,28}\bnot\s+(?:available|accepted|provided|eligible|offered)|"
    r"\bwithout\s+(?:visa\s+)?sponsorship|\b(?:cannot|unable\s+to)\s+sponsor",
    re.I,
)
AUTH_AVAILABILITY = re.compile(
    r"\bsponsorship\s+(?:is\s+)?available|\bvisa sponsorship jobs|"
    r"\bh-?1b\s+(?:visa\s+)?sponsorship|\b(?:will|may)\s+sponsor",
    re.I,
)
AUTH_REQUIREMENT = re.compile(r"\bauthorized to work|\bwork authorization|\blegally authorized", re.I)


def authorization_category(text):
    text = str(text)
    if AUTH_RESTRICTION.search(text):
        return "explicit_restriction"
    if AUTH_REQUIREMENT.search(text):
        return "authorization_requirement"
    if AUTH_AVAILABILITY.search(text):
        return "availability_or_aggregator_mention"
    if AUTH_KEYWORDS.search(text):
        return "ambiguous_mention"
    return "no_title_evidence"


def scan_full_source(path):
    columns = [
        "job_title_short",
        "job_title",
        "job_location",
        "job_via",
        "job_schedule_type",
        "job_work_from_home",
        "job_posted_date",
        "job_country",
        "company_name",
        "job_skills",
    ]
    matches = []
    controls = []
    total_rows = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        chunk = chunk.copy()
        chunk["source_row_idx"] = np.arange(total_rows, total_rows + len(chunk))
        total_rows += len(chunk)
        title = chunk["job_title"].fillna("")
        matched = title.str.contains(AUTH_KEYWORDS, regex=True, na=False)
        selected = chunk[matched].copy()
        selected["authorization_candidate_category"] = selected["job_title"].map(authorization_category)
        matches.append(selected)
        control_pool = chunk[~matched]
        if len(control_pool):
            controls.append(control_pool.sample(n=min(30, len(control_pool)), random_state=2411 + len(controls)))

    evidence = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=columns)
    evidence["normalized_title"] = evidence["job_title"].map(normalize_text)
    evidence["normalized_company"] = evidence["company_name"].fillna("").map(normalize_text)
    evidence = evidence.drop_duplicates(["normalized_title", "normalized_company", "job_location"]).copy()
    evidence["evidence_scope"] = "job title only - advisor verification required"
    evidence["advisor_authorization_label"] = ""
    evidence["review_notes"] = ""

    control = pd.concat(controls, ignore_index=True).drop_duplicates("source_row_idx")
    control = control.sample(n=min(200, len(control)), random_state=2411).copy()
    control["authorization_candidate_category"] = "no_title_evidence"
    control["normalized_title"] = control["job_title"].map(normalize_text)
    control["normalized_company"] = control["company_name"].fillna("").map(normalize_text)
    control["evidence_scope"] = "negative control - title has no authorization keyword"
    control["advisor_authorization_label"] = ""
    control["review_notes"] = ""
    benchmark = pd.concat([evidence, control], ignore_index=True)
    counts = evidence["authorization_candidate_category"].value_counts().to_dict()
    return total_rows, evidence, benchmark, counts


def posting_text_from_raw(frame):
    return (
        "Job title: " + frame["job_title"].fillna("").astype(str)
        + ". Short title: " + frame["job_title_short"].fillna("").astype(str)
        + ". Company: " + frame["company_name"].fillna("").astype(str)
        + ". Location: " + frame["job_location"].fillna("").astype(str)
        + ". Country: " + frame["job_country"].fillna("").astype(str)
        + ". Schedule: " + frame["job_schedule_type"].fillna("").astype(str)
        + ". Skills: " + frame["job_skills"].fillna("").astype(str)
    )


def build_annotation_queue(project, evidence, model):
    validation = project[project["split"].eq("validation")].copy()
    validation["model_predicted_role_fit"] = model.predict(validation["posting_text"].fillna(""))
    decisions = np.asarray(model.decision_function(validation["posting_text"].fillna("")), dtype=float)
    ordered = np.sort(decisions, axis=1)
    validation["model_margin"] = ordered[:, -1] - ordered[:, -2]
    uncertainty_parts = []
    for label in LABELS:
        part = validation[validation["model_predicted_role_fit"].eq(label)].nsmallest(200, "model_margin")
        uncertainty_parts.append(part)
    uncertainty = pd.concat(uncertainty_parts, ignore_index=True)
    uncertainty_queue = pd.DataFrame(
        {
            "queue_source": "active_learning_low_margin",
            "record_id": uncertainty["posting_id"].astype(str),
            "company": uncertainty["company"],
            "role_title": uncertainty["role_title"],
            "location": uncertainty["location"],
            "country": uncertainty["job_country"],
            "posting_text": uncertainty["posting_text"],
            "model_predicted_role_fit": uncertainty["model_predicted_role_fit"],
            "model_margin": uncertainty["model_margin"],
            "weak_role_fit_label": uncertainty["relevance_label"],
            "authorization_candidate_category": "not_selected_for_authorization_text",
        }
    )

    category_order = [
        "explicit_restriction",
        "authorization_requirement",
        "availability_or_aggregator_mention",
        "ambiguous_mention",
    ]
    authorization_parts = []
    remaining = 200
    for category in category_order:
        pool = evidence[evidence["authorization_candidate_category"].eq(category)]
        target = min(50, len(pool), remaining)
        if target:
            authorization_parts.append(pool.sample(n=target, random_state=2411 + len(authorization_parts)))
            remaining -= target
    if remaining:
        used = pd.concat(authorization_parts, ignore_index=True)["source_row_idx"] if authorization_parts else []
        pool = evidence[~evidence["source_row_idx"].isin(used)]
        authorization_parts.append(pool.sample(n=min(remaining, len(pool)), random_state=2499))
    authorization = pd.concat(authorization_parts, ignore_index=True).head(200).copy()
    authorization["posting_text"] = posting_text_from_raw(authorization)
    authorization["model_predicted_role_fit"] = model.predict(authorization["posting_text"])
    decisions = np.asarray(model.decision_function(authorization["posting_text"]), dtype=float)
    ordered = np.sort(decisions, axis=1)
    authorization["model_margin"] = ordered[:, -1] - ordered[:, -2]
    authorization_queue = pd.DataFrame(
        {
            "queue_source": "full_source_authorization_candidate",
            "record_id": "full-" + authorization["source_row_idx"].astype(str),
            "company": authorization["company_name"],
            "role_title": authorization["job_title"],
            "location": authorization["job_location"],
            "country": authorization["job_country"],
            "posting_text": authorization["posting_text"],
            "model_predicted_role_fit": authorization["model_predicted_role_fit"],
            "model_margin": authorization["model_margin"],
            "weak_role_fit_label": "not_available_for_full_source_candidate",
            "authorization_candidate_category": authorization["authorization_candidate_category"],
        }
    )
    queue = pd.concat([uncertainty_queue, authorization_queue], ignore_index=True)
    queue["advisor_role_fit_label"] = ""
    queue["advisor_authorization_label"] = ""
    queue["advisor_action"] = ""
    queue["review_notes"] = ""
    return queue


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


def temporal_validation(project, seed):
    frame = project.copy()
    frame["job_posted_date"] = pd.to_datetime(frame["job_posted_date"], errors="coerce")
    frame = frame.sort_values(["job_posted_date", "posting_id"]).reset_index(drop=True)
    boundary = int(len(frame) * 0.8)
    train = frame.iloc[:boundary].copy()
    test = frame.iloc[boundary:].copy()
    train["text_key"] = train["posting_text"].fillna("").map(normalize_text)
    test["text_key"] = test["posting_text"].fillna("").map(normalize_text)
    train = train.drop_duplicates("text_key", keep="first")
    started = time.time()
    model = make_pipeline(random_state=seed)
    model.fit(train["posting_text"].fillna(""), train["relevance_label"])
    predictions = model.predict(test["posting_text"].fillna(""))
    train_text = set(train["text_key"])
    novel = test[~test["text_key"].isin(train_text)]
    payload = {
        "split_policy": "earliest 80% by job_posted_date for training; newest 20% for testing",
        "train_rows_before_deduplication": boundary,
        "train_rows_after_exact_text_deduplication": len(train),
        "train_date_max": str(train["job_posted_date"].max()),
        "test_date_min": str(test["job_posted_date"].min()),
        "fit_and_evaluation_seconds": round(time.time() - started, 1),
        "full_temporal_test": metric_block(test, predictions),
        "novel_text_temporal_test": metric_block(
            novel, model.predict(novel["posting_text"].fillna(""))
        ),
        "note": "Metrics measure agreement with project weak labels, not advisor ground truth.",
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Audit authorization evidence and build advisor annotation assets.")
    parser.add_argument("--full-source", required=True)
    parser.add_argument("--project-data", default="jobs.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--seed", type=int, default=2411)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    outputs_dir = Path(args.outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    total_rows, evidence, benchmark, counts = scan_full_source(args.full_source)
    benchmark.to_csv(output_dir / "auth.csv", index=False)

    project = pd.read_csv(args.project_data)
    model = joblib.load(args.model)
    queue = build_annotation_queue(project, evidence, model)
    queue.to_csv(output_dir / "review.csv", index=False)

    audit = {
        "source_dataset": "lukebarousse/data_jobs",
        "rows_scanned": total_rows,
        "unique_title_level_authorization_candidates": len(evidence),
        "benchmark_rows_including_controls": len(benchmark),
        "candidate_counts": counts,
        "annotation_queue_rows": len(queue),
        "annotation_queue_composition": queue["queue_source"].value_counts().to_dict(),
        "governance_note": "Title-level phrases are candidate evidence only and require advisor verification.",
    }
    (outputs_dir / "auth.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    temporal = temporal_validation(project, args.seed)
    (outputs_dir / "time.json").write_text(json.dumps(temporal, indent=2), encoding="utf-8")
    print(json.dumps({"authorization_audit": audit, "temporal_validation": temporal}, indent=2))


if __name__ == "__main__":
    main()
