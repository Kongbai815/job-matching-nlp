import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LABELS = ["high_fit", "medium_fit", "low_fit", "unclear"]
TOP_K = 6
DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_REVIEW_POLICY_PATH = Path("models/policy.json")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "will",
    "work", "job", "role", "title", "short", "company", "country", "schedule",
    "remote", "status", "skills", "mention", "false", "true", "not", "student",
    "students", "candidate", "candidates", "posting", "postings",
}


def tokenize(text):
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", str(text))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def feature_terms(text):
    tokens = tokenize(text)
    return tokens + [tokens[i] + "__" + tokens[i + 1] for i in range(len(tokens) - 1)]


def parse_skills_from_text(text):
    match = re.search(r"Skills:\s*(.*?)\.\s*No degree mention:", str(text), flags=re.I)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return []
    return [s.strip().lower() for s in raw.split(";") if s.strip()]


def rubric_scores(text):
    text_l = str(text).lower()
    skills = parse_skills_from_text(text_l)
    if not skills:
        inferred_terms = ["sql", "python", "excel", "tableau", "power bi", "looker", "r"]
        skills = [term for term in inferred_terms if re.search(rf"\b{re.escape(term)}\b", text_l)]
    skill_blob = " ".join(skills)
    msba_terms = {
        "sql", "python", "r", "excel", "tableau", "power bi", "looker", "sas",
        "analytics", "statistics", "forecasting", "dashboard", "powerpoint",
        "business", "a/b",
    }
    msba_skill_count = sum(1 for s in skills if any(term in s for term in msba_terms))
    missing = "skills: ." in text_l or "schedule: ." in text_l or "company: ." in text_l
    country_us = "country: united states" in text_l or "united states" in text_l or re.search(r"\bus\b", text_l) is not None
    analyst_title = any(
        term in text_l
        for term in [
            "short title: data analyst", "short title: business analyst", "business intelligence",
            "analytics analyst", "operations analyst", "data analyst", "business analyst",
        ]
    )
    relevant = any(term in text_l for term in ["analyst", "analytics", "data role", "data scientist", "business intelligence"])
    senior_or_engineering = any(
        term in text_l
        for term in [
            "senior", "principal", "staff", "head ", "director", "manager",
            "short title: data engineer", "data engineer", "machine learning engineer",
            "software", "cloud", "devops", "spark", "hadoop", "kubernetes", "scala",
            "tensorflow", "pytorch", "aws", "azure", "gcp",
        ]
    )
    scores = {label: 0.0 for label in LABELS}
    if missing or len(skills) == 0:
        scores["unclear"] += 3.0
    if any(term in text_l for term in ["sponsorship", "sponsor", "opt", "cpt", "h-1b", "h1b"]) and any(term in text_l for term in ["missing", "does not", "not provide", "without"]):
        scores["unclear"] += 2.0
    if senior_or_engineering:
        scores["low_fit"] += 3.0
    if country_us and analyst_title and msba_skill_count >= 2:
        scores["high_fit"] += 4.0
    if relevant and msba_skill_count >= 2:
        scores["medium_fit"] += 2.0
    if relevant and msba_skill_count < 2:
        scores["unclear"] += 1.5
    if not relevant:
        scores["low_fit"] += 1.0
        scores["unclear"] += 1.0
    if "short title: data scientist" in text_l and not senior_or_engineering:
        scores["medium_fit"] += 1.2
    if country_us and "remote status: remote" in text_l:
        scores["high_fit"] += 0.4
    if any(term in skill_blob for term in ["sql", "tableau", "power bi", "excel"]):
        scores["high_fit"] += 0.3
        scores["medium_fit"] += 0.3
    return scores


def safe_skills(value):
    raw = str(value).strip()
    if raw.lower() in {"", "nan", "none", "null"}:
        return []
    return [s.strip() for s in raw.split(";") if s.strip()]


def parse_search_constraints(query):
    """Extract only constraints that can be verified against structured posting fields."""
    query_l = str(query).lower()
    country = None
    if re.search(r"\b(?:u\.?s\.?|united states)\b", query_l):
        country = "United States"

    remote_mode = "not_specified"
    if re.search(r"\b(?:remote only|must be remote|fully remote|required remote)\b", query_l):
        remote_mode = "required"
    elif re.search(r"\b(?:on[- ]?site only|onsite only|not remote)\b", query_l):
        remote_mode = "excluded"
    elif re.search(r"\b(?:prefer(?:red)? remote|remote preferred)\b", query_l):
        remote_mode = "preferred"
    elif "remote" in query_l:
        remote_mode = "allowed"

    known_skills = [
        "sql", "python", "excel", "tableau", "power bi", "looker", "r",
        "sas", "spark", "aws", "azure", "gcp",
    ]
    skills = [term for term in known_skills if re.search(rf"\b{re.escape(term)}\b", query_l)]
    entry_level = bool(re.search(r"\b(?:entry[- ]level|junior|intern(?:ship)?|student)\b", query_l))
    return {
        "country": country,
        "remote_mode": remote_mode,
        "entry_level": entry_level,
        "requested_skills": skills,
    }


def extract_authorization_evidence(row):
    """Return only explicit authorization language present in the posting."""
    source = " ".join(
        str(row.get(field, ""))
        for field in ("role_title", "posting_text")
        if str(row.get(field, "")).lower() not in {"", "nan", "none", "null"}
    )
    patterns = [
        (r"\bno\s+opt(?:\s*/\s*|\s+)cpt\b", "explicit source text: no OPT/CPT"),
        (r"\bno\s+cpt(?:\s*/\s*|\s+)opt\b", "explicit source text: no CPT/OPT"),
        (r"\bno\s+(?:visa\s+)?sponsorship\b", "explicit source text: no sponsorship"),
        (r"\bsponsorship\s+(?:is\s+)?not\s+(?:available|provided)\b", "explicit source text: sponsorship not available"),
        (r"\b(?:visa\s+)?sponsorship\s+(?:is\s+)?(?:available|provided)\b", "explicit source text: sponsorship available"),
    ]
    for pattern, evidence in patterns:
        if re.search(pattern, source, flags=re.I):
            return evidence
    return "not available in public source"


def missing_structured_fields(text):
    text = str(text)
    fields = {
        "company": r"\bcompany:\s*([^.;]+)",
        "location": r"\blocation:\s*([^.;]+)",
        "country": r"\bcountry:\s*([^.;]+)",
        "skills": r"\bskills:\s*([^.;]+)",
        "schedule": r"\bschedule:\s*([^.;]+)",
    }
    matches = {name: re.search(pattern, text, flags=re.I) for name, pattern in fields.items()}
    if sum(match is not None for match in matches.values()) < 2:
        return []
    missing_values = {"", "unknown", "missing", "not provided", "not available", "n/a", "none", "null"}
    missing = []
    for name, match in matches.items():
        value = match.group(1).strip().lower() if match else ""
        if not match or value in missing_values:
            missing.append(name)
    return missing


def compute_metrics(y_true, y_pred):
    confusion = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1
    per_label = {}
    f1s = []
    correct = 0
    for label in LABELS:
        tp = confusion[label][label]
        correct += tp
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    return {"accuracy": correct / len(y_true), "macro_f1": sum(f1s) / len(f1s), "per_label": per_label, "confusion": confusion}


class JobMatchingSystem:
    def __init__(
        self,
        data_path="jobs.csv",
        top_k=TOP_K,
        model_path=DEFAULT_MODEL_PATH,
        review_policy_path=DEFAULT_REVIEW_POLICY_PATH,
    ):
        self.data_path = Path(data_path)
        if not self.data_path.exists() and self.data_path.name == "jobs.csv":
            self.data_path = Path("jobs.csv")
        self.model_path = Path(model_path) if model_path else None
        if self.model_path and not self.model_path.exists() and self.model_path.name == DEFAULT_MODEL_PATH.name:
            candidate = Path(__file__).resolve().parents[1] / DEFAULT_MODEL_PATH
            self.model_path = candidate if candidate.exists() else self.model_path
        self.classifier = joblib.load(self.model_path) if self.model_path and self.model_path.exists() else None
        self.review_policy_path = Path(review_policy_path) if review_policy_path else None
        if self.review_policy_path and not self.review_policy_path.exists() and self.review_policy_path.name == DEFAULT_REVIEW_POLICY_PATH.name:
            candidate = Path(__file__).resolve().parents[1] / DEFAULT_REVIEW_POLICY_PATH
            self.review_policy_path = candidate if candidate.exists() else self.review_policy_path
        self.review_policy = (
            json.loads(self.review_policy_path.read_text(encoding="utf-8"))
            if self.review_policy_path and self.review_policy_path.exists()
            else {
                "policy_name": "fallback uniform margin policy",
                "per_label_margin_thresholds": {label: 0.35 for label in LABELS},
                "metric_scope": "fallback only",
            }
        )
        self.classifier_name = (
            "word+character TF-IDF LinearSVC (77,135 deduplicated training rows)"
            if self.classifier is not None
            else "transparent rubric fallback"
        )
        self.top_k = top_k
        self.df = pd.read_csv(self.data_path)
        self.train_df = self.df[self.df["split"].eq("train")].reset_index(drop=True)
        self.records = self.train_df.to_dict("records")
        self.index = defaultdict(list)
        self.doc_norm = []
        self.df_counts = Counter()
        self._build_index()

    def _build_index(self):
        doc_terms = []
        for text in self.train_df["posting_text"]:
            terms = set(feature_terms(text))
            doc_terms.append(terms)
            self.df_counts.update(terms)
        n_docs = len(doc_terms)
        self.idf = {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in self.df_counts.items()}
        for doc_id, terms in enumerate(doc_terms):
            selected = [term for term in terms if 3 <= self.df_counts[term] <= 9000]
            norm = math.sqrt(sum(self.idf[term] ** 2 for term in selected)) or 1.0
            self.doc_norm.append(norm)
            for term in selected:
                self.index[term].append(doc_id)

    def retrieve(self, query, top_k=None):
        top_k = top_k or self.top_k
        constraints = parse_search_constraints(query)
        query_terms = list(set(feature_terms(query)))
        query_terms = [term for term in query_terms if term in self.index and 3 <= self.df_counts[term] <= 9000]
        query_terms.sort(key=lambda term: self.idf.get(term, 0.0), reverse=True)
        query_terms = query_terms[:28]
        scores = Counter()
        for term in query_terms:
            weight = self.idf[term]
            for doc_id in self.index[term]:
                scores[doc_id] += (weight * weight) / float(self.doc_norm[doc_id])
        candidates = []
        for doc_id, lexical_score in scores.most_common(max(200, top_k * 30)):
            row = self.records[doc_id]
            if constraints["country"] and str(row.get("job_country", "")) != constraints["country"]:
                continue
            is_remote = self._is_remote(row)
            if constraints["remote_mode"] == "required" and not is_remote:
                continue
            if constraints["remote_mode"] == "excluded" and is_remote:
                continue
            if constraints["entry_level"] and re.search(
                r"\b(?:senior|principal|staff|director|manager|lead)\b",
                str(row.get("role_title", "")),
                flags=re.I,
            ):
                continue
            adjusted_score = lexical_score + self._constraint_adjustment(query, row, constraints)
            candidates.append((doc_id, adjusted_score))
        candidates.sort(key=lambda item: item[1], reverse=True)

        ranked = []
        seen = set()
        for doc_id, score in candidates:
            row = self.records[doc_id]
            dedupe_key = (
                re.sub(r"\W+", " ", str(row.get("company", "")).lower()).strip(),
                re.sub(r"\W+", " ", str(row.get("role_title", "")).lower()).strip(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            ranked.append((doc_id, score))
            if len(ranked) == top_k:
                break
        return [self._format_record(rank, doc_id, score) for rank, (doc_id, score) in enumerate(ranked, start=1)]

    @staticmethod
    def _is_remote(row):
        title_l = str(row.get("role_title", "")).lower()
        location_l = str(row.get("location", "")).lower()
        value = row.get("job_work_from_home")
        structured_remote = bool(value) and str(value).lower() not in {"false", "0", "nan", "none"}
        return structured_remote or location_l == "anywhere" or "remote" in title_l

    @classmethod
    def _constraint_adjustment(cls, query, row, constraints=None):
        query_l = str(query).lower()
        title_l = str(row.get("role_title", "")).lower()
        constraints = constraints or parse_search_constraints(query)
        adjustment = 0.0
        if constraints["country"]:
            adjustment += 0.8
        if constraints["remote_mode"] == "preferred" and cls._is_remote(row):
            adjustment += 0.5
        if constraints["entry_level"]:
            if any(term in title_l for term in ["senior", "principal", "staff", "director", "manager"]):
                adjustment -= 2.0
        row_skills = {skill.lower() for skill in safe_skills(row.get("job_skills", ""))}
        adjustment += 0.2 * sum(skill in row_skills for skill in constraints["requested_skills"])
        return adjustment

    def _format_record(self, rank, doc_id, score):
        row = self.records[doc_id]
        skills = safe_skills(row.get("job_skills", ""))
        model_label, _, model_margin = self.predict_details(row.get("posting_text", ""))
        return {
            "rank": rank,
            "score": float(score),
            "posting_id": row.get("posting_id"),
            "company": row.get("company"),
            "role_title": row.get("role_title"),
            "location": row.get("location"),
            "job_country": row.get("job_country"),
            "work_from_home": self._is_remote(row),
            "skills": skills[:8],
            "grounded_fit_label": row.get("relevance_label"),
            "model_fit_label": model_label,
            "model_margin": model_margin,
            "label_reason": row.get("label_reason"),
            "authorization_evidence": extract_authorization_evidence(row),
        }

    def predict_label(self, text):
        label, scores, _ = self.predict_details(text)
        return label, scores

    def predict_details(self, text):
        if self.classifier is not None:
            label = str(self.classifier.predict([str(text)])[0])
            decision = np.asarray(self.classifier.decision_function([str(text)])[0], dtype=float)
            classes = [str(value) for value in self.classifier.classes_]
            scores = {name: float(value) for name, value in zip(classes, decision)}
            ordered = np.sort(decision)
            margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1])
            return label, scores, margin
        scores = rubric_scores(text)
        ordered = sorted(scores.values(), reverse=True)
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
        return max(LABELS, key=lambda label: scores[label]), scores, margin

    def run(self, query, top_k=None, input_mode="auto"):
        if input_mode == "auto":
            input_mode = "posting" if re.search(r"\b(?:job title|short title|company|country|skills):", str(query), flags=re.I) else "search"
        if input_mode not in {"search", "posting"}:
            raise ValueError("input_mode must be 'search', 'posting', or 'auto'.")
        retrieved = self.retrieve(query, top_k=top_k)
        applied_search_constraints = parse_search_constraints(query) if input_mode == "search" else {}
        if input_mode == "posting":
            model_predicted_label, scores, prediction_margin = self.predict_details(query)
            missing_fields = missing_structured_fields(query)
            predicted_label = "unclear" if len(missing_fields) >= 2 else model_predicted_label
            input_authorization_evidence = extract_authorization_evidence({"posting_text": query})
        else:
            model_predicted_label = "not_applicable_search_query"
            predicted_label, scores, prediction_margin = model_predicted_label, {}, None
            missing_fields = []
            input_authorization_evidence = "not available in public source"
        review_threshold = (
            float(self.review_policy.get("per_label_margin_thresholds", {}).get(model_predicted_label, 0.35))
            if input_mode == "posting"
            else None
        )
        low_confidence_review = bool(
            input_mode == "posting"
            and prediction_margin is not None
            and prediction_margin < review_threshold
        )
        priority = {"high_fit": 0, "medium_fit": 1, "unclear": 2, "low_fit": 3}
        ordered = sorted(retrieved, key=lambda item: (priority.get(item["model_fit_label"], 9), -item["score"]))
        for rank, item in enumerate(ordered, start=1):
            item["rank"] = rank
        review_count = sum(1 for item in ordered if item["model_fit_label"] in {"high_fit", "medium_fit"})
        explicit_restriction = input_authorization_evidence in {
            "explicit source text: no OPT/CPT",
            "explicit source text: no CPT/OPT",
            "explicit source text: no sponsorship",
            "explicit source text: sponsorship not available",
        }
        policy_reasons = []
        if input_mode == "search":
            policy_reasons.append("search results require advisor selection")
        if predicted_label in {"low_fit", "unclear"}:
            policy_reasons.append("low-fit or unclear result requires review")
        if missing_fields:
            policy_reasons.append("structured posting fields are missing")
        if explicit_restriction:
            policy_reasons.append("explicit authorization restriction requires review")
        if low_confidence_review:
            policy_reasons.append("model margin is below the calibrated class threshold")
        if explicit_restriction:
            headline = "Role fit and work authorization conflict: hold this posting for advisor verification."
        elif input_mode == "posting" and predicted_label in {"low_fit", "unclear"}:
            headline = "Route this query to advisor review; retrieved postings provide evidence but do not override the triage result."
        elif review_count:
            headline = f"Review {review_count} retrieved postings first; the trained classifier found analytics role-fit signals."
        else:
            headline = "No strong match found; route this query to manual advisor review."
        explicit_authorization_count = sum(
            item["authorization_evidence"] != "not available in public source"
            for item in ordered
        )
        if explicit_authorization_count:
            grounding_caveat = (
                "Authorization language is reported only when it appears explicitly in retrieved source text. "
                "An advisor must verify the original posting; no eligibility is inferred."
            )
        else:
            grounding_caveat = (
                "Do not infer CPT, OPT, or sponsorship from this public dataset. "
                "Authorization evidence must be reviewed separately."
            )
        return {
            "input": query,
            "input_mode": input_mode,
            "model_predicted_label": model_predicted_label,
            "predicted_posting_label": predicted_label,
            "classifier_name": self.classifier_name,
            "classifier_scores": scores,
            "prediction_margin": prediction_margin,
            "review_margin_threshold": review_threshold,
            "low_confidence_review": low_confidence_review,
            "review_policy_name": self.review_policy.get("policy_name"),
            "missing_core_fields": missing_fields,
            "input_authorization_evidence": input_authorization_evidence,
            "applied_search_constraints": applied_search_constraints,
            "policy_reasons": policy_reasons,
            "review_required": (
                input_mode == "search"
                or predicted_label in {"low_fit", "unclear"}
                or explicit_restriction
                or low_confidence_review
            ),
            "headline": headline,
            "recommended_action": (
                "Hold - explicit authorization restriction requires advisor verification"
                if explicit_restriction
                else "Investigate - advisor review required"
                if predicted_label in {"low_fit", "unclear"} or low_confidence_review
                else "Advisor review before forwarding"
            ),
            "grounding_caveat": grounding_caveat,
            "explicit_authorization_evidence_count": explicit_authorization_count,
            "retrieved_evidence": ordered,
        }
