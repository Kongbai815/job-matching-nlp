# Final Workflow Casebook

## entry_level_analytics_search

- Input mode: `search`
- Business use: Find US analytics opportunities for advisor review
- Posting label: `not_applicable_search_query`
- Review required: `True`
- Action: Advisor review before forwarding
- Authorization evidence: not available in public source
- Top evidence:
  - `HF-469758` - Data Analyst at HireMatch (United States), model label `high_fit`.
  - `HF-250208` - Data Analyst (US REMOTE) at LeanTaaS (United States), model label `high_fit`.
  - `HF-219271` - Data Analyst at HMP Global (United States), model label `high_fit`.

## senior_engineering_posting

- Input mode: `posting`
- Business use: Hold a role that is too senior and engineering-heavy
- Posting label: `low_fit`
- Review required: `True`
- Action: Advisor review before forwarding
- Authorization evidence: not available in public source
- Top evidence:
  - `HF-590539` - Data Engineer at Charlie's Produce (United States), model label `unclear`.
  - `HF-1197` - Machine Translation Data Engineer at MindSource (United States), model label `unclear`.
  - `HF-535995` - GCP Data Engineer at Adaminfotech (United States), model label `low_fit`.

## authorization_restriction_posting

- Input mode: `posting`
- Business use: Separate strong role fit from explicit authorization restrictions
- Posting label: `high_fit`
- Review required: `True`
- Action: Hold - explicit authorization restriction requires advisor verification
- Authorization evidence: explicit source text: no OPT/CPT
- Top evidence:
  - `HF-734288` - Jr. Data Analyst (No OPT/CPT) at Winorbit Technology (United States), model label `high_fit`.
  - `HF-779383` - Data Analyst at Fervorly (United States), model label `high_fit`.
  - `HF-164738` - Jr. Data Analyst at Daman (United States), model label `high_fit`.

## thin_metadata_posting

- Input mode: `posting`
- Business use: Escalate an incomplete posting
- Posting label: `unclear`
- Review required: `True`
- Action: Advisor review before forwarding
- Authorization evidence: not available in public source
- Top evidence:
  - `HF-671521` - Summer Data Analytics Internship at ALBEMARLE (United States), model label `high_fit`.
  - `HF-745488` - Data Analytics Internship at ReedTMS Logistics (United States), model label `high_fit`.
  - `HF-112682` - Compliance Analytics Internship at Wise (Singapore), model label `medium_fit`.

## Governance Check

- Cases with retrieved evidence: 4/4.
- Explicit restrictions correctly forced review: 1.
- Unsupported authorization claims: 0.
