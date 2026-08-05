# Governance and Risk Appendix

## 1. Unsupported Authorization Claims

**Affected users:** international students and career advisors.

**Failure mode:** the system infers CPT, OPT, H-1B, or sponsorship eligibility from a posting that does not contain reliable authorization text.

**Mitigation:** authorization is separated from role fit; the generator surfaces only explicit phrases and otherwise uses a missing-evidence caveat. An explicit negative phrase forces `review_required=true` and a Hold action even when role fit is high. Four saved workflow cases contain 0 unsupported authorization claims.

## 2. Weak-Label Bias

**Affected users:** candidates whose viable roles contain engineering vocabulary, unusual titles, or incomplete metadata.

**Failure mode:** a model can reproduce transparent weak-label rules very accurately without matching human advisor judgment.

**Mitigation:** report the scores as weak-label agreement, expose retrieved evidence, prohibit automatic rejection, and collect 500-1,000 de-identified advisor labels through stratified active learning before a real deployment.

## 3. Duplicate and Organization Leakage

**Failure mode:** random splitting places duplicate descriptions or the same employer patterns in both train and validation, inflating model scores.

**Mitigation:** exact-text duplicates are removed before final training. We also report novel-text and unseen-company validation slices; both remain near 0.98 macro F1. A future test set should be time-based and advisor-labeled.

## 4. Privacy

**Affected users:** students and employers in future Career Center data.

**Failure mode:** resumes, emails, or internal posting notes expose personal or confidential information.

**Mitigation:** the current prototype uses public posting data only. A pilot must de-identify student text, minimize stored fields, restrict access, and define retention periods before ingestion.

## 5. Automation Harm and Misuse

**Failure mode:** staff treat the label as a hiring decision or eligibility determination.

**Mitigation:** frame the product as prioritization support, keep the advisor as the final decision maker, log model evidence and overrides, and audit outcomes by label and relevant user group.

## Known Limitations

- Labels are weak labels, not human-reviewed ground truth.
- The retriever is lexical and can miss paraphrases.
- All 100,000 rows have `auth_signal=not_available_in_source`; authorization extraction therefore has only case-based checks, not a representative labeled evaluation.
- Business value such as time saved has not yet been measured in a live advisor workflow.
