# MSBA Job Matching NLP System

Production-ready capstone prototype for ISBA 2411. The system helps a graduate business career advisor screen job opportunities for MSBA international students. It takes a real advisor query or raw posting text, retrieves similar real job postings, predicts a triage label, and returns a grounded recommendation with cited evidence.

## Problem Statement

Career advisors receive many postings, emails, and links. They must decide what to forward, hold, or investigate. For international students, role fit and work authorization are separate questions: a posting can match MSBA skills while still lacking reliable CPT, OPT, H-1B, or sponsorship evidence.

## Data

- Source: `lukebarousse/data_jobs`
- Original records: **785,741**
- Stratified project sample: **100,000**
- Train/validation split: **80,000 / 20,000**
- Labels: `high_fit`, `medium_fit`, `low_fit`, `unclear`
- Evaluation note: labels are transparent weak labels, not final human advisor judgments

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
```

## Architecture

1. **Input:** advisor query or raw job text.
2. **Retrieval:** sparse inverted index over 80,000 real training postings.
3. **Decision:** transparent role-fit rubric predicts one of four triage labels.
4. **Grounded output:** the response cites retrieved title, company, location, skills, label, and label reason.
5. **Governance gate:** explicit authorization language is surfaced; missing text triggers a caveat and human review.

## Evaluation

| System | Evaluation rows | Accuracy | Macro F1 | Role in project |
| --- | ---: | ---: | ---: | --- |
| Milestone 2 TF-IDF baseline | 20,000 | 0.793 | 0.785 | Static baseline |
| Milestone 3 true PEFT LoRA | 4,000 | 0.875 | 0.873 | Best pure classifier |
| Final end-to-end system | 4,000 | 0.800 | 0.801 | Retrieval, evidence, and guardrails |

On the same balanced 4,000-row subset, the final prototype improves macro F1 from 0.791 to 0.801. The cross-milestone table is descriptive because the M2 and M3 experiments use different modeling and evaluation setups.

Retrieval success is 100.0%, support hit@5 is 90.0%, and the four-case governance check found 0 unsupported authorization claims while preserving 1 explicit source statement.

## Findings and Recommendation

- High-fit triage is strong: F1 is 0.960.
- The main errors are `medium_fit -> low_fit` and `unclear -> low_fit`; borderline cases need human review.
- True PEFT LoRA is the best classifier, while the final prototype is the stronger decision-support system because it returns evidence and enforces the authorization caveat.
- Recommended deployment: first-pass prioritization with advisor confirmation, followed by a pilot on de-identified Career Center postings.

## Reproducibility and Deliverables

- M2-M4 completed notebooks and archival PDFs are stored at the repository root.
- Final casebook: `outputs/final_casebook_outputs.json` and `docs/final_casebook.md`.
- Governance appendix: `docs/governance_risk_appendix.md`.
- Final technical summary: `docs/Final_Technical_Summary.pdf`.
- Slide deck: `slides/MSBA_Job_Matching_Final_Deck.pptx` and `.pdf`.
- Recorded demo: `demo/final_demo_video.mp4`.
