# MSBA Job Matching NLP System

Production-ready capstone prototype for ISBA 2411. The system helps a graduate business career advisor screen job opportunities for MSBA international students. It takes a real advisor query or raw posting text, retrieves similar real job postings, predicts a triage label, and returns a grounded recommendation with cited evidence.

## Problem Statement

Career advisors receive many job postings, emails, and links. For MSBA international students, the advisor must quickly decide whether an opportunity is worth forwarding, needs review, or is likely a poor fit. This is hard because job descriptions vary widely, analytics fit is distributed across title, skills, location, and schedule fields, and authorization language is often missing.

## Data

- Source dataset: `lukebarousse/data_jobs`
- Original public records: **785,741**
- Project sample: **100,000**
- Train/validation split: **80,000 / 20,000**
- Labels: `high_fit`, `medium_fit`, `low_fit`, `unclear`

Labels are weak labels created from transparent advisor-screening rules. They are useful for baseline development, but they are not final human career-advisor judgments.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The repo is self-contained with `data_jobs_msba_project_sample_100k.csv`; the CLI also accepts a `data/` subfolder if you choose to organize the CSV there.

## Usage

Run the normal demo:

```bash
python -m msba_job_matcher.app --input-json demo/sample_query.json --output outputs/final_demo_output.json
```

Run the edge case:

```bash
python -m msba_job_matcher.app --input-json demo/edge_case_query.json --output outputs/final_edge_case_output.json
```

Run evaluation:

```bash
python -m msba_job_matcher.evaluate --sample-per-label 1000 --output outputs/final_evaluation_results.json
```

## Architecture

1. Input: advisor query or raw job text.
2. Retrieval: sparse inverted index over 80,000 real training postings.
3. Decision: transparent role-fit rubric predicts `high_fit`, `medium_fit`, `low_fit`, or `unclear`.
4. Grounded output: retrieved postings supply title, company, location, skills, and label evidence.
5. Governance guardrail: the system never infers CPT, OPT, H-1B, or sponsorship from the public dataset.

## Evaluation

| System | Eval rows | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Milestone 2 TF-IDF baseline | 20,000 | 0.793 | 0.785 |
| Milestone 3 true PEFT LoRA | 4,000 | 0.875 | 0.873 |
| Final end-to-end system | 4,000 | 0.800 | 0.801 |

The final system is not only a classifier: it returns grounded evidence and a governance caveat. Retrieval success is 100.0%, and support hit@5 in Milestone 4 was 90.0%.

The true PEFT LoRA experiment is the strongest pure classifier, but the final system is the better capstone prototype because it also returns evidence, prevents unsupported authorization claims, and keeps the advisor in the review loop.

## Findings and Recommendation

The project is feasible as an advisor decision-support tool. The strongest business value is reducing first-pass screening time while keeping human review for ambiguous authorization and low-confidence cases. The recommended next step is to add de-identified Career Center postings that include CPT/OPT/sponsorship language, then evaluate authorization extraction separately.

## Final Deliverables

- Demo video: `demo/final_demo_video.mp4`
- Slide deck PDF: `slides/MSBA_Job_Matching_Final_Deck.pdf`
- Governance appendix: `docs/governance_risk_appendix.md`
- Architecture: `docs/architecture.md`
- Evaluation summary: `docs/evaluation_summary.md`
