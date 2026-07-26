# ISBA 2411 NLP Job-Matching Project

This repository contains the MSBA job-matching NLP project milestone notebooks and supporting data.

## Milestone 2 - Baseline & Representation

- `MSBA_Job_Matching_Milestone2.ipynb`
- `MSBA_Job_Matching_Milestone2_Notebook_Copy.pdf`
- TF-IDF validation accuracy: 0.7926
- TF-IDF macro F1: 0.7850

## Milestone 3 - Model Adaptation

- `MSBA_Job_Matching_Milestone3_Model_Adaptation.ipynb`
- `MSBA_Job_Matching_Milestone3_Notebook_Copy.pdf`
- `MSBA_Job_Matching_Milestone3_500_Word_Analysis.txt`
- `milestone3_adaptation_results.json`
- Zero-shot macro F1: 0.8147
- Few-shot macro F1: 0.8285
- LoRA-style low-rank adapter macro F1: 0.8652

## Data

- Source dataset: `lukebarousse/data_jobs`
- Original source rows: 785,741
- Project sample: 100,000 rows
- Split: 80,000 train / 20,000 validation
- Labels: `high_fit`, `medium_fit`, `low_fit`, `unclear`

The CSV is included so the notebook can run without relying on a local-only file path.
