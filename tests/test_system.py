import json
import unittest
from pathlib import Path

import pandas as pd

from msba_job_matcher.core import JobMatchingSystem


ROOT = Path(__file__).resolve().parents[1]


class SystemSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = JobMatchingSystem(
            data_path=ROOT / "data_jobs_msba_project_sample_100k.csv",
            model_path=ROOT / "models" / "job_fit_tfidf_svc.joblib",
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
        metrics = json.loads((ROOT / "models" / "job_fit_tfidf_svc_metrics.json").read_text(encoding="utf-8"))
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
            (ROOT / "outputs" / "temporal_validation_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(temporal["full_temporal_test"]["rows"], 20000)
        self.assertGreater(temporal["full_temporal_test"]["macro_f1"], 0.97)
        self.assertGreater(temporal["novel_text_temporal_test"]["macro_f1"], 0.97)

    def test_full_source_audit_and_annotation_assets(self):
        audit = json.loads(
            (ROOT / "outputs" / "authorization_evidence_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(audit["rows_scanned"], 785741)
        self.assertEqual(audit["unique_title_level_authorization_candidates"], 230)
        self.assertEqual(audit["benchmark_rows_including_controls"], 430)
        self.assertEqual(audit["annotation_queue_rows"], 1000)
        self.assertEqual(
            audit["annotation_queue_composition"],
            {"active_learning_low_margin": 800, "full_source_authorization_candidate": 200},
        )

        queue = pd.read_csv(ROOT / "data" / "advisor_annotation_queue_1000.csv", keep_default_na=False)
        benchmark = pd.read_csv(
            ROOT / "data" / "authorization_evidence_benchmark.csv", keep_default_na=False
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


if __name__ == "__main__":
    unittest.main()
