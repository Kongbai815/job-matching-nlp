# Architecture

## End-to-End Flow

1. The advisor selects `search` mode for a request or `posting` mode for a job description; `auto` detects structured posting fields.
2. A sparse unigram/bigram index retrieves candidates from 80,000 real postings, then applies location/seniority boosts and duplicate suppression.
3. A word+character TF-IDF LinearSVC scores postings. It was trained on 77,135 exact-text-deduplicated rows.
4. A policy layer changes incomplete structured postings to `unclear` and forces review when explicit authorization restrictions appear.
5. The generator assembles a recommendation from retrieved title, company, location, skills, model label, and source label reason.
6. The advisor receives a review decision, source evidence, missing-field list, model margin, and authorization caveat.

## Component Boundaries

- `msba_job_matcher/core.py`: mode routing, retrieval, classifier inference, policy gates, and grounded response.
- `msba_job_matcher/app.py`: command-line input/output interface.
- `msba_job_matcher/train.py`: reproducible deduplicated model training and robustness evaluation.
- `msba_job_matcher/evaluate.py`: balanced validation evaluation.
- `msba_job_matcher/casebook.py`: four reproducible workflow and governance cases.
- `demo/*.json`: reproducible normal and edge inputs.
- `outputs/*.json`: saved system runs, evaluation metrics, and casebook evidence.

## Design Decision

The final system uses deterministic, evidence-bound generation. The trained classifier is used for role fit, but policy gates remain deterministic because the public data cannot validate immigration eligibility. Search intent, posting classification, and authorization review are therefore separate decisions.
