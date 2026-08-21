import json
import unittest
from pathlib import Path

import pandas as pd

from jobmatch.core import JobMatchingSystem
from jobmatch.batch import process_batch_frame


ROOT = Path(__file__).resolve().parents[1]


class SystemSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = JobMatchingSystem(
            data_path=ROOT / "jobs.csv",
            model_path=ROOT / "models" / "model.joblib",
        )

    def test_search_mode_retrieves_us_analytics_jobs(self):
        result = self.system.run(
            "Find entry-level US data analyst roles with SQL, Python, Tableau, or Power BI.",
            input_mode="search",
        )
        self.assertEqual(result["input_mode"], "search")
        self.assertEqual(result["predicted_posting_label"], "not_applicable_search_query")
        self.assertTrue(result["retrieved_evidence"])
        self.assertEqual(
            [item["rank"] for item in result["retrieved_evidence"]],
            list(range(1, len(result["retrieved_evidence"]) + 1)),
        )
        self.assertTrue(all(item["job_country"] == "United States" for item in result["retrieved_evidence"][:5]))
        self.assertEqual(result["applied_search_constraints"]["country"], "United States")

    def test_remote_only_is_a_hard_search_constraint(self):
        result = self.system.run(
            "Find remote only US data analyst roles with SQL and Tableau.",
            input_mode="search",
        )
        self.assertTrue(result["retrieved_evidence"])
        self.assertEqual(result["applied_search_constraints"]["remote_mode"], "required")
        self.assertTrue(all(item["work_from_home"] for item in result["retrieved_evidence"]))
        self.assertTrue(all(item["job_country"] == "United States" for item in result["retrieved_evidence"]))

    def test_explicit_authorization_restriction_forces_review(self):
        text = (
            "Job title: Jr. Data Analyst (No OPT/CPT). Short title: Data Analyst. "
            "Company: Example Staffing. Location: Austin, TX. Country: United States. "
            "Schedule: Full-time. Skills: SQL; Python; Excel."
        )
        result = self.system.run(text, input_mode="posting")
        self.assertEqual(result["predicted_posting_label"], "high_fit")
        self.assertEqual(result["input_authorization_evidence"], "explicit source text: no OPT/CPT")
        self.assertTrue(result["review_required"])
        self.assertIn("Hold", result["recommended_action"])

    def test_saved_metrics_include_leakage_controlled_slices(self):
        metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
        self.assertGreater(metrics["evaluation"]["fixed_4k"]["macro_f1"], 0.98)
        self.assertGreater(metrics["evaluation"]["full_20k_novel_text"]["macro_f1"], 0.97)
        self.assertGreater(metrics["evaluation"]["full_20k_unseen_company"]["macro_f1"], 0.97)

    def test_missing_structured_fields_route_to_unclear(self):
        text = "Job title: Analytics Internship. Company: unknown. Location: unknown. Skills: Excel."
        result = self.system.run(text, input_mode="posting")
        self.assertEqual(result["predicted_posting_label"], "unclear")
        self.assertIn("company", result["missing_core_fields"])
        self.assertTrue(result["review_required"])

    def test_time_based_validation_is_saved_and_strong(self):
        temporal = json.loads(
            (ROOT / "outputs" / "time.json").read_text(encoding="utf-8")
        )
        self.assertEqual(temporal["full_temporal_test"]["rows"], 20000)
        self.assertGreater(temporal["full_temporal_test"]["macro_f1"], 0.97)
        self.assertGreater(temporal["novel_text_temporal_test"]["macro_f1"], 0.97)

    def test_full_source_audit_and_annotation_assets(self):
        audit = json.loads(
            (ROOT / "outputs" / "auth.json").read_text(encoding="utf-8")
        )
        self.assertEqual(audit["rows_scanned"], 785741)
        self.assertEqual(audit["unique_title_level_authorization_candidates"], 230)
        self.assertEqual(audit["benchmark_rows_including_controls"], 430)
        self.assertEqual(audit["annotation_queue_rows"], 1000)
        self.assertEqual(
            audit["annotation_queue_composition"],
            {"active_learning_low_margin": 800, "full_source_authorization_candidate": 200},
        )

        queue = pd.read_csv(ROOT / "data" / "review.csv", keep_default_na=False)
        benchmark = pd.read_csv(
            ROOT / "data" / "auth.csv", keep_default_na=False
        )
        self.assertEqual(len(queue), 1000)
        self.assertEqual(len(benchmark), 430)
        human_fields = [
            "advisor_role_fit_label",
            "advisor_authorization_label",
            "advisor_action",
            "review_notes",
        ]
        self.assertTrue((queue[human_fields] == "").all().all())

    def test_calibrated_review_policy_and_low_margin_gate(self):
        policy = json.loads((ROOT / "models" / "policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["calibration_rows"], 10000)
        self.assertEqual(policy["audit_rows"], 10000)
        self.assertGreater(policy["independent_audit"]["coverage"], 0.90)
        self.assertGreater(policy["independent_audit"]["selective_accuracy"], 0.99)

        queue = pd.read_csv(ROOT / "data" / "review.csv")
        low_margin_text = queue.sort_values("model_margin").iloc[0]["posting_text"]
        result = self.system.run(low_margin_text, input_mode="posting")
        self.assertTrue(result["low_confidence_review"])
        self.assertTrue(result["review_required"])
        self.assertIn("calibrated", " ".join(result["policy_reasons"]))

    def test_batch_review_reuses_one_system_and_returns_audit_columns(self):
        frame = pd.DataFrame(
            [
                {"mode": "search", "text": "Find remote only US data analyst roles with SQL."},
                {
                    "mode": "posting",
                    "text": (
                        "Job title: Senior Data Engineer. Short title: Data Engineer. "
                        "Company: Example. Location: Seattle, WA. Country: United States. "
                        "Schedule: Full-time. Skills: Spark; AWS; Kubernetes."
                    ),
                },
            ]
        )
        completed = process_batch_frame(frame, self.system)
        self.assertEqual(len(completed), 2)
        self.assertEqual(completed.loc[0, "input_mode"], "search")
        self.assertEqual(completed.loc[1, "predicted_posting_label"], "low_fit")
        self.assertTrue(completed["result_json"].str.contains("review_policy_name").all())


if __name__ == "__main__":
    unittest.main()
