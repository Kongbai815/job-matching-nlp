# Production Classifier

`job_fit_tfidf_svc.joblib` is the final role-fit decision model used by the M4/M5 system. It combines word and character TF-IDF features with a linear support-vector classifier.

- Training source: 80,000 project training rows.
- Exact-text duplicates removed before fitting: 77,135 retained rows.
- Fixed balanced validation set: 4,000 rows.
- Full held-out validation set: 20,000 rows.
- Rebuild command: `python -m msba_job_matcher.train`.

The model was selected after comparing word-only, character-only, and combined representations. It is more accurate, faster to train, and simpler to reproduce than the 2,000-row M3 LoRA experiment. Its scores measure agreement with transparent weak labels, not agreement with human career-advisor judgments.
