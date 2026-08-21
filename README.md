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
python -m msba_job_matcher.app --input-json demo/query.json --output outputs/final_demo_output.json
python -m msba_job_matcher.app --input-json demo/edge.json --output outputs/final_edge_case_output.json
python -m msba_job_matcher.final_demo
python -m msba_job_matcher.evaluate --sample-per-label 1000 --output outputs/final_evaluation_results.json
python -m msba_job_matcher.casebook
python -m msba_job_matcher.calibrate_review_policy
python -m msba_job_matcher.evaluate_search_constraints
python -m msba_job_matcher.batch --input demo/batch.csv --output outputs/batch_review_results.csv
python -m msba_job_matcher.audit_and_annotation --full-source path/to/data_jobs.csv
```

Rebuild the committed classifier with `python -m msba_job_matcher.train`.

## Architecture

1. **Mode routing:** separate search requests from job postings so a search sentence is never treated as a posting label.
2. **Retrieval:** sparse index over 80,000 real postings, with explicit country/remote/entry-level constraints, skill-aware ranking, and duplicate suppression.
3. **Decision:** word+character TF-IDF LinearSVC trained on 77,135 deduplicated rows.
4. **Policy gates:** class-specific margins are calibrated on 10,000 rows and audited on a separate 10,000; missing fields and explicit authorization restrictions remain deterministic overrides.
5. **Grounded output:** the response cites retrieved title, company, location, skills, model label, and source label reason.
6. **Operations:** single-input JSON and batch CSV entry points reuse the same trained system and emit auditable policy reasons.

## Evaluation

| System | Evaluation rows | Accuracy | Macro F1 | Role in project |
| --- | ---: | ---: | ---: | --- |
| Milestone 2 TF-IDF baseline | 20,000 | 0.793 | 0.785 | Static baseline |
| Milestone 3 true PEFT LoRA | 4,000 | 0.875 | 0.873 | Adaptation experiment |
| Final trained classifier | 4,000 | 0.984 | 0.984 | Decision layer in the end-to-end system |

On the same balanced 4,000-row subset, macro F1 improves from 0.791 to 0.984. A leakage-controlled audit remains strong: 0.979 macro F1 on 18,672 validation texts not seen in training and 0.980 on 5,570 rows from companies not seen in training. These are still weak-label metrics, not advisor-ground-truth accuracy.

A separate time-based experiment trains on the earliest 80% of postings and evaluates on the newest 20%. It reaches **0.976 macro F1** on 20,000 rows and **0.976** on 19,611 novel-text rows, reducing the risk that the random split hides temporal drift.

Retrieval success is 100.0%, support hit@5 is 90.0%, and the four-case governance check found 0 unsupported authorization claims while preserving 1 explicit source statement.

The calibrated review policy targets 99.5% weak-label precision on a 10,000-row calibration half. On the independent 10,000-row audit half, it retains **91.63% coverage** at **99.47% selective accuracy** and routes 8.37% of cases to mandatory review. A six-query search audit returns 36 results with **0 hard-constraint violations** and **0 duplicate company-title pairs**.

## Findings and Recommendation

- All four class F1 scores exceed 0.969 on the fixed validation set.
- Training-size experiments show that using the available labels matters more than a larger model: macro F1 rises from 0.898 at 2,000 rows to 0.984 at 80,000 rows.
- The final linear model outperforms the 2,000-row LoRA experiment while training in under a minute and producing a 3.8 MB artifact.
- A calibrated class-specific review policy replaces the arbitrary global margin threshold and exposes the threshold and reason for each decision.
- Country, remote-only/on-site-only, and entry-level constraints are enforced before ranking rather than treated only as score boosts.
- The batch CSV command converts the prototype from a one-query demo into a repeatable review workflow.
- The repository now includes a 1,000-row advisor annotation queue and a 430-row authorization benchmark built from a full-source audit. Human-review fields remain blank by design; model-generated labels are not substituted for human ground truth.
- Recommended deployment: first-pass prioritization with advisor confirmation, followed by a pilot on de-identified Career Center postings.

## Recorded Demo

The **3:15** silent screen recording is available at [`demo/demo.mp4`](demo/demo.mp4). It records the actual command `python -m msba_job_matcher.final_demo --recording`, including live initialization of the 100,000-row dataset, a normal search request, grounded retrieval output, an explicit `NO OPT/CPT` edge case, the batch review workflow, and measured evaluation. English headings and explanations remain on screen long enough for the recording to be understood without audio.

The demonstrated outputs are generated during the recording and committed under `outputs/`. The edge case intentionally preserves a `high_fit` role prediction while forcing `review_required=true` and a Hold action, showing that role fit never overrides explicit authorization evidence. Run the same sequence quickly with `python -m msba_job_matcher.final_demo`, or use `--recording` to add readable pauses between scenes.

## Reproducibility and Deliverables

- M2-M4 completed notebooks and archival PDFs are stored at the repository root.
- Final casebook: `outputs/final_casebook_outputs.json` and `docs/final_casebook.md`.
- Model artifact and audit: `models/job_fit_tfidf_svc.joblib` and `models/job_fit_tfidf_svc_metrics.json`.
- Review policy and search audit: `models/review_policy.json` and `outputs/search_constraint_audit.json`.
- Batch example: `demo/batch.csv` and `outputs/batch_review_results.csv`.
- Annotation assets: `data/advisor_annotation_queue_1000.csv`, `data/authorization_evidence_benchmark.csv`, and `docs/annotation_guide.md`.
- Full-source and temporal audits: `outputs/authorization_evidence_audit.json` and `outputs/temporal_validation_results.json`.
- Governance appendix: `docs/governance_risk_appendix.md`.
- Final technical summary: `docs/Final_Technical_Summary.pdf`.
- Slide deck: `slides/MSBA_Job_Matching_Final_Deck.pptx` and `.pdf`.
- Recorded demo: `demo/demo.mp4`, a silent screen recording of the reproducible real-run command.
