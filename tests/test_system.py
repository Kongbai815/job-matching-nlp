import json
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
