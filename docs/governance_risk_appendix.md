# Governance and Risk Appendix

## 1. Unsupported Authorization Claims

**Affected users:** international students and career advisors.

**Failure mode:** the system infers CPT, OPT, H-1B, or sponsorship eligibility from a posting that does not contain reliable authorization text.

**Mitigation:** authorization is separated from role fit; the generator surfaces only explicit phrases and otherwise uses a missing-evidence caveat. An explicit negative phrase forces `review_required=true` and a Hold action even when role fit is high. Four saved workflow cases contain 0 unsupported authorization claims.

## 2. Weak-Label Bias

**Affected users:** candidates whose viable roles contain engineering vocabulary, unusual titles, or incomplete metadata.

**Failure mode:** a model can reproduce transparent weak-label rules very accurately without matching human advisor judgment.

**Mitigation:** report the scores as weak-label agreement, expose retrieved evidence, prohibit automatic rejection, and use the included 1,000-row active-learning queue for de-identified advisor review before a real deployment.

## 3. Duplicate and Organization Leakage

**Failure mode:** random splitting places duplicate descriptions or the same employer patterns in both train and validation, inflating model scores.

**Mitigation:** exact-text duplicates are removed before final training. Novel-text and unseen-company slices remain near 0.98 macro F1. A completed time-based split reaches 0.976 macro F1 on the newest 20,000 postings; the human-reviewed queue remains the external validity test.

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
- All 100,000 project rows have `auth_signal=not_available_in_source`. A full-source scan finds 230 unique title-level authorization candidates and packages a 430-row candidate/control benchmark, but the advisor labels remain intentionally blank.
- Business value such as time saved has not yet been measured in a live advisor workflow.

## Full-Source Authorization Audit

The project scans all 785,741 original rows. It finds 230 unique job-title-level authorization candidates: 35 explicit restrictions, 47 availability or sponsorship-source mentions, and 148 ambiguous mentions. A 430-row benchmark adds 200 no-keyword controls. These phrases are evidence candidates, not verified eligibility labels.

The 1,000-row advisor queue combines 800 low-margin role-fit cases with 200 full-source authorization candidates. Human fields are intentionally blank so the evaluated model cannot manufacture its own ground truth.
