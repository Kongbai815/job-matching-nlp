# Architecture

## End-to-End Flow

1. The advisor supplies a free-text search request or raw posting text.
2. Text is tokenized into unigrams and bigrams; common workflow words are removed.
3. A sparse inverted index retrieves the top six comparable postings from 80,000 training records.
4. A transparent rubric predicts `high_fit`, `medium_fit`, `low_fit`, or `unclear`.
5. The generator assembles a recommendation from retrieved title, company, location, skills, label, and reason fields.
6. Explicit authorization phrases are surfaced; missing evidence triggers a caveat and advisor escalation.

## Component Boundaries

- `msba_job_matcher/core.py`: retrieval, role-fit rubric, grounded response, and metrics.
- `msba_job_matcher/app.py`: command-line input/output interface.
- `msba_job_matcher/evaluate.py`: balanced validation evaluation.
- `demo/*.json`: reproducible normal and edge inputs.
- `outputs/*.json`: saved system runs, evaluation metrics, and casebook evidence.

## Design Decision

The final system uses deterministic, evidence-bound generation. This is intentionally conservative: the public dataset can support role-fit evidence but cannot reliably support work-authorization claims. The architecture therefore separates role-fit triage from authorization review.
