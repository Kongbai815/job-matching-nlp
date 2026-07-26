import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

LABELS = ["high_fit", "medium_fit", "low_fit", "unclear"]
TOP_K = 6

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
    def __init__(self, data_path="data/data_jobs_msba_project_sample_100k.csv", top_k=TOP_K):
        self.data_path = Path(data_path)
        if not self.data_path.exists() and self.data_path.name == "data_jobs_msba_project_sample_100k.csv":
            self.data_path = Path("data_jobs_msba_project_sample_100k.csv")
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
        query_terms = list(set(feature_terms(query)))
        query_terms = [term for term in query_terms if term in self.index and 3 <= self.df_counts[term] <= 9000]
        query_terms.sort(key=lambda term: self.idf.get(term, 0.0), reverse=True)
        query_terms = query_terms[:28]
        scores = Counter()
        for term in query_terms:
            weight = self.idf[term]
            for doc_id in self.index[term]:
                scores[doc_id] += (weight * weight) / float(self.doc_norm[doc_id])
        ranked = scores.most_common(top_k)
        return [self._format_record(rank, doc_id, score) for rank, (doc_id, score) in enumerate(ranked, start=1)]

    def _format_record(self, rank, doc_id, score):
        row = self.records[doc_id]
        skills = safe_skills(row.get("job_skills", ""))
        return {
            "rank": rank,
            "score": float(score),
            "posting_id": row.get("posting_id"),
            "company": row.get("company"),
            "role_title": row.get("role_title"),
            "location": row.get("location"),
            "job_country": row.get("job_country"),
            "skills": skills[:8],
            "grounded_fit_label": row.get("relevance_label"),
            "label_reason": row.get("label_reason"),
            "authorization_evidence": "not available in public source",
        }

    def predict_label(self, text):
        scores = rubric_scores(text)
        return max(LABELS, key=lambda label: scores[label]), scores

    def run(self, query, top_k=None):
        retrieved = self.retrieve(query, top_k=top_k)
        predicted_label, scores = self.predict_label(query)
        priority = {"high_fit": 0, "medium_fit": 1, "unclear": 2, "low_fit": 3}
        ordered = sorted(retrieved, key=lambda item: (priority.get(item["grounded_fit_label"], 9), -item["score"]))
        review_count = sum(1 for item in ordered if item["grounded_fit_label"] in {"high_fit", "medium_fit"})
        headline = (
            f"Review {review_count} retrieved postings first; they have analytics role-fit signals."
            if review_count
            else "No strong match found; route this query to manual advisor review."
        )
        return {
            "input": query,
            "predicted_query_label": predicted_label,
            "rubric_scores": scores,
            "headline": headline,
            "recommended_action": "Advisor review before forwarding",
            "grounding_caveat": "Do not infer CPT, OPT, or sponsorship from this public dataset. Authorization evidence must be reviewed separately.",
            "retrieved_evidence": ordered,
        }
