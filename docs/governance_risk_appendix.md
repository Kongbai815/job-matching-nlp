# Governance and Risk Appendix

## 1. Unsupported Authorization Claims

**Affected users:** international students and career advisors.

**Failure mode:** the system infers CPT, OPT, H-1B, or sponsorship eligibility from a posting that does not contain reliable authorization text.

**Mitigation:** authorization is separated from role fit; the generator surfaces only explicit phrases and otherwise uses a missing-evidence caveat; the advisor must verify eligibility. In four saved workflow cases covering 20 retrieved records, the check found 0 unsupported authorization claims.

## 2. Weak-Label Bias

**Affected users:** candidates whose viable roles contain engineering vocabulary, unusual titles, or incomplete metadata.

**Failure mode:** transparent weak-label rules over-penalize a posting and push a medium or unclear opportunity into low fit.

**Mitigation:** report per-label performance, expose retrieved evidence, prohibit automatic rejection, and collect de-identified advisor corrections. The largest observed confusions are medium_fit -> low_fit (380) and unclear -> low_fit (254).

## 3. Privacy

**Affected users:** students and employers in future Career Center data.

**Failure mode:** resumes, emails, or internal posting notes expose personal or confidential information.

**Mitigation:** the current prototype uses public posting data only. A pilot must de-identify student text, minimize stored fields, restrict access, and define retention periods before ingestion.

## 4. Automation Harm and Misuse

**Failure mode:** staff treat the label as a hiring decision or eligibility determination.

**Mitigation:** frame the product as prioritization support, keep the advisor as the final decision maker, log model evidence and overrides, and audit outcomes by label and relevant user group.

## Known Limitations

- Labels are weak labels, not human-reviewed ground truth.
- The retriever is lexical and can miss paraphrases.
- The public dataset cannot validate authorization extraction.
- Business value such as time saved has not yet been measured in a live advisor workflow.
