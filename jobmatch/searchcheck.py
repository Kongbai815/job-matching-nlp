import argparse
import json
import re
from pathlib import Path

from .core import JobMatchingSystem


SEARCH_CASES = [
    "Find entry-level US data analyst roles with SQL and Python.",
    "Find remote only US data analyst roles with SQL and Tableau.",
    "Find on-site only US business analyst roles with Excel and Power BI.",
    "Find remote only analytics roles with Python and Tableau.",
    "Find US business intelligence roles with SQL and Power BI.",
    "Find entry-level US analytics internships with Excel and dashboards.",
]


def constraint_violations(result):
    constraints = result["applied_search_constraints"]
    violations = []
    for item in result["retrieved_evidence"]:
        reasons = []
        if constraints["country"] and item["job_country"] != constraints["country"]:
            reasons.append("country")
        if constraints["remote_mode"] == "required" and not item["work_from_home"]:
            reasons.append("remote_required")
        if constraints["remote_mode"] == "excluded" and item["work_from_home"]:
            reasons.append("remote_excluded")
        if constraints["entry_level"] and re.search(
            r"\b(?:senior|principal|staff|director|manager|lead)\b",
            str(item["role_title"]),
            flags=re.I,
        ):
            reasons.append("entry_level")
        if reasons:
            violations.append({"posting_id": item["posting_id"], "reasons": reasons})
    return violations


def main():
    parser = argparse.ArgumentParser(description="Evaluate deterministic search-constraint compliance.")
    parser.add_argument("--data", default="jobs.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--review-policy", default="models/policy.json")
    parser.add_argument("--output", default="outputs/search-audit.json")
    args = parser.parse_args()

    system = JobMatchingSystem(
        data_path=args.data,
        model_path=args.model,
        review_policy_path=args.review_policy,
    )
    cases = []
    total_results = 0
    total_violations = 0
    duplicate_pairs = 0
    for query in SEARCH_CASES:
        result = system.run(query, input_mode="search")
        violations = constraint_violations(result)
        pairs = [
            (item["company"].lower(), item["role_title"].lower())
            for item in result["retrieved_evidence"]
        ]
        duplicates = len(pairs) - len(set(pairs))
        total_results += len(pairs)
        total_violations += len(violations)
        duplicate_pairs += duplicates
        cases.append(
            {
                "query": query,
                "applied_constraints": result["applied_search_constraints"],
                "returned_results": len(pairs),
                "constraint_violations": violations,
                "duplicate_company_title_pairs": duplicates,
            }
        )
    payload = {
        "queries": len(cases),
        "returned_results": total_results,
        "hard_constraint_violations": total_violations,
        "hard_constraint_compliance": 1 - total_violations / total_results if total_results else 0.0,
        "duplicate_company_title_pairs": duplicate_pairs,
        "cases": cases,
        "scope_note": "This audit measures deterministic constraint compliance, not human relevance judgment.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
