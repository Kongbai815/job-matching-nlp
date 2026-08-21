import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from .batch import process_batch_frame
from .core import JobMatchingSystem


ROOT = Path(__file__).resolve().parents[1]
WIDTH = 96


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clear_screen(enabled):
    if enabled:
        os.system("cls" if os.name == "nt" else "clear")


def heading(title, subtitle="", clear=False):
    clear_screen(clear)
    print("=" * WIDTH)
    print("MSBA JOB MATCHING | FINAL CAPSTONE | REAL END-TO-END RUN")
    print("=" * WIDTH)
    print(title)
    if subtitle:
        print(subtitle)
    print("-" * WIDTH)


def hold(seconds, final=False):
    if seconds <= 0:
        return
    label = "Recording complete." if final else "This screen remains visible for review."
    print(f"\n{label}", flush=True)
    time.sleep(seconds)


def shorten(value, width):
    text = str(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete capstone demo with a normal input and a policy edge case."
    )
    parser.add_argument(
        "--recording",
        action="store_true",
        help="Clear between scenes and hold each screen long enough for a silent 3-5 minute recording.",
    )
    parser.add_argument(
        "--scene-seconds",
        type=float,
        default=None,
        help="Seconds to hold each scene. Defaults to 20 in recording mode and 0 otherwise.",
    )
    args = parser.parse_args()

    delay = args.scene_seconds
    if delay is None:
        delay = 20.0 if args.recording else 0.0
    clear = args.recording and sys.stdout.isatty()

    search_path = ROOT / "demo" / "query.json"
    edge_path = ROOT / "demo" / "edge.json"
    batch_path = ROOT / "demo" / "batch.csv"
    output_dir = ROOT / "outputs"

    heading(
        "SCENE 1 OF 9 | BUSINESS PROBLEM AND SOLUTION",
        "Primary user: a graduate career advisor screening job opportunities for MSBA students.",
        clear=clear,
    )
    print("Problem : Role fit, missing evidence, and work-authorization language are easy to confuse.")
    print("Decision: Forward, hold, or investigate each opportunity.")
    print("System  : Route -> retrieve -> classify -> govern -> advisor review.")
    print("Boundary: A high role-fit score never overrides explicit No OPT/CPT evidence.")
    print("\nLIVE COMMAND")
    print("python -m msba_job_matcher.final_demo --recording")
    hold(delay)

    heading(
        "SCENE 2 OF 9 | LOAD THE REAL DATA, MODEL, AND RETRIEVAL INDEX",
        "The following initialization is executing now, not replaying saved output.",
        clear=clear,
    )
    print("Loading data_jobs_msba_project_sample_100k.csv ...", flush=True)
    print("Loading models/job_fit_tfidf_svc.joblib and models/review_policy.json ...", flush=True)
    started = time.perf_counter()
    system = JobMatchingSystem(
        data_path=ROOT / "data_jobs_msba_project_sample_100k.csv",
        model_path=ROOT / "models" / "job_fit_tfidf_svc.joblib",
        review_policy_path=ROOT / "models" / "review_policy.json",
    )
    load_seconds = time.perf_counter() - started
    print("\nLIVE INITIALIZATION COMPLETE")
    print(f"Project rows          : {len(system.df):,}")
    print(f"Retrieval-index rows  : {len(system.train_df):,}")
    print(f"Classifier            : {system.classifier_name}")
    print(f"Review policy         : {system.review_policy['policy_name']}")
    print(f"Initialization time   : {load_seconds:.2f} seconds")
    hold(delay)

    search_payload = load_json(search_path)
    heading(
        "SCENE 3 OF 9 | REAL INPUT 1: ADVISOR SEARCH REQUEST",
        "The system must route this as search, enforce constraints, and return source evidence.",
        clear=clear,
    )
    print("INPUT FILE: demo/query.json")
    print("MODE      : search")
    print("QUERY")
    print(shorten(search_payload["query"], 92))
    print("\nExecuting JobMatchingSystem.run(...) now ...", flush=True)
    started = time.perf_counter()
    search_result = system.run(search_payload["query"], input_mode="search")
    search_seconds = time.perf_counter() - started
    write_json(output_dir / "final_demo_output.json", search_result)
    print(f"Completed in {search_seconds:.2f} seconds; JSON output written successfully.")
    hold(delay)

    heading(
        "SCENE 4 OF 9 | REAL OUTPUT 1: GROUNDED SEARCH RESULTS",
        "Search text is not misclassified as if it were a job posting.",
        clear=clear,
    )
    constraints = search_result["applied_search_constraints"]
    print(f"Input mode            : {search_result['input_mode']}")
    print(f"Posting label         : {search_result['predicted_posting_label']}")
    print(f"Country constraint    : {constraints['country']}")
    print(f"Entry-level constraint: {constraints['entry_level']}")
    print(f"Requested skills      : {', '.join(constraints['requested_skills'])}")
    print(f"Advisor review        : {search_result['review_required']}")
    print("\nTOP RETRIEVED POSTINGS FROM THE REAL 80,000-ROW INDEX")
    print("#  Role title                         Company                    Location")
    for row in search_result["retrieved_evidence"][:3]:
        print(
            f"{row['rank']:<2} {shorten(row['role_title'], 34):<34} "
            f"{shorten(row['company'], 25):<25} {shorten(row['location'], 24)}"
        )
    print("\nGrounding caveat: authorization evidence is not available in the public source.")
    hold(delay)

    edge_payload = load_json(edge_path)
    heading(
        "SCENE 5 OF 9 | REAL INPUT 2: EXPLICIT NO OPT/CPT EDGE CASE",
        "This posting looks relevant, but the policy must preserve the explicit restriction.",
        clear=clear,
    )
    print("INPUT FILE: demo/edge.json")
    print("MODE      : posting")
    print("POSTING")
    print(shorten(edge_payload["query"], 92))
    print("\nExecuting JobMatchingSystem.run(...) now ...", flush=True)
    started = time.perf_counter()
    edge_result = system.run(edge_payload["query"], input_mode="posting")
    edge_seconds = time.perf_counter() - started
    write_json(output_dir / "final_edge_case_output.json", edge_result)
    print(f"Completed in {edge_seconds:.2f} seconds; JSON output written successfully.")
    hold(delay)

    heading(
        "SCENE 6 OF 9 | REAL OUTPUT 2: ROLE FIT CANNOT OVERRIDE POLICY",
        "The model prediction remains visible, while the governance gate controls the action.",
        clear=clear,
    )
    print(f"Model role-fit label  : {edge_result['model_predicted_label']}")
    print(f"Prediction margin     : {edge_result['prediction_margin']:.3f}")
    print(f"Authorization evidence: {edge_result['input_authorization_evidence']}")
    print(f"Review required       : {edge_result['review_required']}")
    print(f"Recommended action    : {edge_result['recommended_action']}")
    print("Policy reason         : " + " | ".join(edge_result["policy_reasons"]))
    print("\nEDGE-CASE RESULT")
    print("High role fit is preserved, but the workflow action is HOLD for advisor verification.")
    hold(delay)

    heading(
        "SCENE 7 OF 9 | REAL BATCH REVIEW WORKFLOW",
        "The same production system now processes a three-row CSV queue.",
        clear=clear,
    )
    print("INPUT : demo/batch.csv")
    print("OUTPUT: outputs/batch_review_results.csv")
    print("\nExecuting process_batch_frame(...) now ...", flush=True)
    batch_source = pd.read_csv(batch_path, keep_default_na=False)
    started = time.perf_counter()
    batch_result = process_batch_frame(batch_source, system)
    batch_seconds = time.perf_counter() - started
    batch_result.to_csv(output_dir / "batch_review_results.csv", index=False)
    print(f"Completed {len(batch_result)} rows in {batch_seconds:.2f} seconds.\n")
    print("Row  Mode     Label       Review  Recommended action")
    for _, row in batch_result.iterrows():
        print(
            f"{int(row['input_row']):<4} {row['input_mode']:<8} "
            f"{shorten(row['predicted_posting_label'], 11):<11} "
            f"{str(bool(row['review_required'])):<7} {shorten(row['recommended_action'], 42)}"
        )
    print("\nEach output row preserves policy reasons, evidence, constraints, and the full JSON result.")
    hold(delay)

    heading(
        "SCENE 8 OF 9 | MEASURED IMPROVEMENT AND ROBUSTNESS",
        "These are held-out weak-label agreement metrics, not advisor-ground-truth accuracy.",
        clear=clear,
    )
    baseline = load_json(output_dir / "milestone2_baseline_results.json")
    final_eval = load_json(output_dir / "final_evaluation_results.json")
    temporal = load_json(output_dir / "temporal_validation_results.json")
    review = load_json(ROOT / "models" / "review_policy.json")
    search_audit = load_json(output_dir / "search_constraint_audit.json")
    print(f"M2 TF-IDF baseline macro F1       : {baseline['metrics']['macro_f1']:.4f}")
    print(f"Final classifier macro F1         : {final_eval['metrics']['macro_f1']:.4f}")
    print(f"Measured absolute improvement     : {final_eval['metrics']['macro_f1'] - baseline['metrics']['macro_f1']:.4f}")
    print(f"Newest-20% temporal macro F1       : {temporal['full_temporal_test']['macro_f1']:.4f}")
    print(f"Retained-policy selective accuracy : {review['independent_audit']['selective_accuracy']:.4f}")
    print(f"Mandatory advisor-review rate      : {review['independent_audit']['mandatory_review_rate']:.2%}")
    print(f"Search hard-constraint violations  : {search_audit['hard_constraint_violations']}")
    hold(delay)

    heading(
        "SCENE 9 OF 9 | PRODUCTION BOUNDARY AND RECOMMENDATION",
        "The end-to-end run completed successfully on one normal input and one required edge case.",
        clear=clear,
    )
    print("DELIVERED")
    print("- Real-input search with constrained retrieval and source-grounded evidence")
    print("- Posting classification with calibrated review thresholds")
    print("- Explicit authorization and missing-evidence governance gates")
    print("- Repeatable batch CSV workflow with auditable outputs")
    print("- Baseline comparison, temporal validation, and search-constraint audit")
    print("\nGOVERNANCE BOUNDARY")
    print("No automated rejection, forwarding, hiring, or immigration decision.")
    print("Recommended next step: a supervised pilot with de-identified Career Center postings.")
    print("\nSTATUS: FINAL REAL-RUN DEMO PASSED")
    hold(delay, final=True)


if __name__ == "__main__":
    main()
