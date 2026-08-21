# Architecture

## End-to-End Flow

1. The advisor selects `search` mode for a request or `posting` mode for a job description; `auto` detects structured posting fields.
2. A sparse unigram/bigram index retrieves candidates from 80,000 real postings. Country, remote-only/on-site-only, and entry-level constraints are enforced before skill-aware ranking and duplicate suppression.
3. A word+character TF-IDF LinearSVC scores postings. It was trained on 77,135 exact-text-deduplicated rows.
4. A class-specific margin policy, calibrated on 10,000 rows and audited on a separate 10,000, routes uncertain predictions to mandatory review. Incomplete fields and explicit authorization restrictions are separate deterministic overrides.
5. The generator assembles a recommendation from retrieved title, company, location, skills, model label, and source label reason.
6. The advisor receives a review decision, source evidence, missing-field list, model margin, and authorization caveat.

## Component Boundaries

- `jobmatch/core.py`: mode routing, retrieval, classifier inference, policy gates, and grounded response.
- `jobmatch/app.py`: command-line input/output interface.
- `jobmatch/train.py`: reproducible deduplicated model training and robustness evaluation.
- `jobmatch/evaluate.py`: balanced validation evaluation.
- `jobmatch/calibrate.py`: reproducible class-specific review thresholds and held-out selective audit.
- `jobmatch/searchcheck.py`: six-query hard-constraint and duplicate audit.
- `jobmatch/batch.py`: batch CSV review using one loaded retrieval/model instance.
- `jobmatch/cases.py`: four reproducible workflow and governance cases.
- `jobmatch/annotate.py`: full-source authorization scan, time-based evaluation, and reproducible advisor queue generation.
- `demo/*.json`: reproducible normal and edge inputs.
- `data/*.csv`: the 1,000-row advisor queue and 430-row authorization candidate/control benchmark.
- `outputs/*.json`: saved system runs, random and temporal evaluation metrics, audit counts, and casebook evidence.

## Design Decision

The final system uses deterministic, evidence-bound generation. The trained classifier supplies role fit and a margin; the calibrated policy determines whether uncertainty requires mandatory review. Missing fields and authorization phrases remain deterministic because title-level evidence is not verified immigration eligibility. Search intent, posting classification, uncertainty review, and authorization review are separate decisions.
