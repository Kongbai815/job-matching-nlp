# MSBA Job Matching NLP System

Production-ready capstone prototype for ISBA 2411. The system helps a graduate business career advisor screen job opportunities for MSBA international students. It supports two explicit modes: a `search` request retrieves and ranks real postings, while a `posting` input receives a trained role-fit label, comparable evidence, and governance checks.

## Problem Statement

Career advisors receive many postings, emails, and links. They must decide what to forward, hold, or investigate. For international students, role fit and work authorization are separate questions: a posting can match MSBA skills while still lacking reliable CPT, OPT, H-1B, or sponsorship evidence.

## Data

- Source: `lukebarousse/data_jobs`
- Original records: **785,741**
- Stratified project sample: **100,000**
- Train/validation split: **80,000 / 20,000**
- Labels: `high_fit`, `medium_fit`, `low_fit`, `unclear`
- Evaluation note: labels are transparent weak labels, not final human advisor judgments
- Production training rows after exact-text deduplication: **77,135**
- Full-source authorization audit: **785,741 rows scanned**, yielding **230 unique title-level candidates**
- Human-review package: **1,000 rows** (800 low-margin role-fit cases + 200 authorization candidates)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The repository includes `data_jobs_msba_project_sample_100k.csv`. The CLI also accepts the same file under `data/`.

## Usage

```bash
python -m msba_job_matcher.app --input-json demo/sample_query.json --output outputs/final_demo_output.json
python -m msba_job_matcher.app --input-json demo/edge_case_query.json --output outputs/final_edge_case_output.json
python -m msba_job_matcher.evaluate --sample-per-label 1000 --output outputs/final_evaluation_results.json
python -m msba_job_matcher.casebook
python -m msba_job_matcher.audit_and_annotation --full-source path/to/data_jobs.csv
```

Rebuild the committed classifier with `python -m msba_job_matcher.train`.

## Architecture

1. **Mode routing:** separate search requests from job postings so a search sentence is never treated as a posting label.
2. **Retrieval:** sparse index over 80,000 real postings, with query-constraint boosts and duplicate suppression.
3. **Decision:** word+character TF-IDF LinearSVC trained on 77,135 deduplicated rows.
4. **Policy gates:** missing core fields route to `unclear`; explicit authorization restrictions force review.
5. **Grounded output:** the response cites retrieved title, company, location, skills, model label, and source label reason.

## Evaluation

| System | Evaluation rows | Accuracy | Macro F1 | Role in project |
| --- | ---: | ---: | ---: | --- |
| Milestone 2 TF-IDF baseline | 20,000 | 0.793 | 0.785 | Static baseline |
| Milestone 3 true PEFT LoRA | 4,000 | 0.875 | 0.873 | Best pure classifier |
| Final trained classifier | 4,000 | 0.984 | 0.984 | Decision layer in the end-to-end system |

On the same balanced 4,000-row subset, macro F1 improves from 0.791 to 0.984. A leakage-controlled audit remains strong: 0.979 macro F1 on 18,672 validation texts not seen in training and 0.980 on 5,570 rows from companies not seen in training. These are still weak-label metrics, not advisor-ground-truth accuracy.

A separate time-based experiment trains on the earliest 80% of postings and evaluates on the newest 20%. It reaches **0.976 macro F1** on 20,000 rows and **0.976** on 19,611 novel-text rows, reducing the risk that the random split hides temporal drift.

Retrieval success is 100.0%, support hit@5 is 90.0%, and the four-case governance check found 0 unsupported authorization claims while preserving 1 explicit source statement.

## Findings and Recommendation

- All four class F1 scores exceed 0.969 on the fixed validation set.
- Training-size experiments show that using the available labels matters more than a larger model: macro F1 rises from 0.898 at 2,000 rows to 0.984 at 80,000 rows.
- The final linear model outperforms the 2,000-row LoRA experiment while training in under a minute and producing a 3.8 MB artifact.
- The repository now includes a 1,000-row advisor annotation queue and a 430-row authorization benchmark built from a full-source audit. Human-review fields remain blank by design; model-generated labels are not substituted for human ground truth.
- Recommended deployment: first-pass prioritization with advisor confirmation, followed by a pilot on de-identified Career Center postings.

## Reproducibility and Deliverables

- M2-M4 completed notebooks and archival PDFs are stored at the repository root.
- Final casebook: `outputs/final_casebook_outputs.json` and `docs/final_casebook.md`.
- Model artifact and audit: `models/job_fit_tfidf_svc.joblib` and `models/job_fit_tfidf_svc_metrics.json`.
- Annotation assets: `data/advisor_annotation_queue_1000.csv`, `data/authorization_evidence_benchmark.csv`, and `docs/annotation_guide.md`.
- Full-source and temporal audits: `outputs/authorization_evidence_audit.json` and `outputs/temporal_validation_results.json`.
- Governance appendix: `docs/governance_risk_appendix.md`.
- Final technical summary: `docs/Final_Technical_Summary.pdf`.
- Slide deck: `slides/MSBA_Job_Matching_Final_Deck.pptx` and `.pdf`.
- Recorded demo: `demo/final_demo_video.mp4`.
