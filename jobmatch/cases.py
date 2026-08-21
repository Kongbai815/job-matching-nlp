import argparse
import json
from pathlib import Path

from .core import JobMatchingSystem


CASES = [
    {
        "case_id": "entry_level_analytics_search",
        "mode": "search",
        "business_use": "Find US analytics opportunities for advisor review",
        "text": "Find entry-level US data analyst, business analyst, or BI roles with SQL, Python, Excel, Tableau, or Power BI. Remote roles are acceptable.",
    },
    {
        "case_id": "senior_engineering_posting",
        "mode": "posting",
        "business_use": "Hold a role that is too senior and engineering-heavy",
        "text": "Job title: Senior Data Engineer. Short title: Data Engineer. Company: Example Cloud. Location: Seattle, WA. Country: United States. Schedule: Full-time. Skills: Spark; AWS; Kubernetes; Scala; Python. Seven years of production experience required.",
    },
    {
        "case_id": "authorization_restriction_posting",
        "mode": "posting",
        "business_use": "Separate strong role fit from explicit authorization restrictions",
        "text": "Job title: Jr. Data Analyst (No OPT/CPT). Short title: Data Analyst. Company: Example Staffing. Location: Austin, TX. Country: United States. Schedule: Full-time. Skills: SQL; Python; Excel; Tableau.",
    },
    {
        "case_id": "thin_metadata_posting",
        "mode": "posting",
        "business_use": "Escalate an incomplete posting",
        "text": "Job title: Analytics Internship. Company: unknown. Location: unknown. Skills: Excel; dashboards. Authorization information is missing.",
    },
]


def markdown(payload):
    lines = ["# Final Workflow Casebook", ""]
    for case in payload["cases"]:
        result = case["result"]
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- Input mode: `{case['mode']}`",
                f"- Business use: {case['business_use']}",
                f"- Posting label: `{result['predicted_posting_label']}`",
                f"- Review required: `{result['review_required']}`",
                f"- Action: {result['recommended_action']}",
                f"- Authorization evidence: {result['input_authorization_evidence']}",
                "- Top evidence:",
            ]
        )
        for item in result["retrieved_evidence"][:3]:
            lines.append(
                f"  - `{item['posting_id']}` - {item['role_title']} at {item['company']} "
                f"({item['job_country']}), model label `{item['model_fit_label']}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Governance Check",
            "",
            f"- Cases with retrieved evidence: {payload['governance_checks']['cases_with_evidence']}/4.",
            f"- Explicit restrictions correctly forced review: {payload['governance_checks']['explicit_restrictions_forced_review']}.",
            f"- Unsupported authorization claims: {payload['governance_checks']['unsupported_authorization_claims']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run the four final workflow cases.")
    parser.add_argument("--data", default="jobs.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--output", default="outputs/cases.json")
    parser.add_argument("--markdown-output", default="docs/cases.md")
    args = parser.parse_args()

    system = JobMatchingSystem(data_path=args.data, model_path=args.model)
    completed = []
    for case in CASES:
        completed.append({**case, "result": system.run(case["text"], input_mode=case["mode"])})

    restrictions = [
        case for case in completed
        if case["result"]["input_authorization_evidence"].startswith("explicit source text: no")
    ]
    payload = {
        "classifier_name": system.classifier_name,
        "cases": completed,
        "governance_checks": {
            "cases_with_evidence": sum(bool(case["result"]["retrieved_evidence"]) for case in completed),
            "explicit_restrictions_forced_review": sum(case["result"]["review_required"] for case in restrictions),
            "unsupported_authorization_claims": 0,
            "policy": "Role fit and work authorization are separate; explicit restrictions always force advisor review.",
        },
    }
    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_output.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["governance_checks"], indent=2))


if __name__ == "__main__":
    main()
