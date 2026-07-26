# Governance and Risk Appendix

## Risk 1: Authorization hallucination

The public dataset does not contain reliable CPT, OPT, H-1B, or sponsorship text. The system must not infer authorization eligibility. Mitigation: every output includes a caveat that authorization evidence is missing unless source text explicitly provides it.

## Risk 2: Weak-label overconfidence

Current labels are rule-derived, not human-reviewed. Mitigation: the output is framed as advisor support, and recommendations require human review before forwarding.

## Risk 3: Bias toward certain titles and geographies

Title and country rules may favor US analyst titles and miss international or nonstandard analytics roles. Mitigation: report performance by label and collect advisor corrections.

## Risk 4: Privacy

This capstone uses public postings only. If Career Center data is added, de-identify student and employer contact information before training or evaluation.

## Failure modes

Missing skills can produce `unclear`; senior technical roles can be over-filtered; authorization details are not available in the public source. All are routed to advisor review rather than automated forwarding.
