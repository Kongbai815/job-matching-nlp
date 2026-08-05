# Final Workflow Casebook

These four saved runs demonstrate the normal workflow plus expected escalation paths.

## Entry Level Analytics

- Business use: Forward candidates after advisor review
- Predicted query label: `high_fit`
- Headline: Review 6 retrieved postings first; they have analytics role-fit signals.
- Grounding caveat: Do not infer CPT, OPT, or sponsorship from this public dataset. Authorization evidence must be reviewed separately.

Top retrieved evidence:

- `HF-469758` - Data Analyst at HireMatch (Anywhere), `high_fit`, score 8.054.
- `HF-178197` - Data Analyst II, Bay Area - Remote at Astreya (Anywhere), `high_fit`, score 7.951.
- `HF-250208` - Data Analyst (US REMOTE) at LeanTaaS (Anywhere), `high_fit`, score 6.858.

## Senior Engineering Filter

- Business use: Hold a role that is too senior or engineering-heavy
- Predicted query label: `low_fit`
- Headline: Review 2 retrieved postings first; they have analytics role-fit signals.
- Grounding caveat: Do not infer CPT, OPT, or sponsorship from this public dataset. Authorization evidence must be reviewed separately.

Top retrieved evidence:

- `HF-672427` - Adidas Recruitment 2023 - 2+Years Experience Required -  Analyst Post at Adidas (Anywhere), `medium_fit`, score 6.482.
- `HF-49084` - Technical Data Analyst (6+ Years Experience) at HARRISS CONSULTANCY AND ENTERPRISE SOLUTIONS (Hyderabad, Telangana, India), `medium_fit`, score 6.366.
- `HF-470734` - ML Engineer/Data Scientist 3+ years of experience - Contract to Hire at Upwork (Anywhere), `unclear`, score 7.359.

## Authorization Missing

- Business use: Escalate missing CPT/OPT or sponsorship evidence
- Predicted query label: `unclear`
- Headline: Review 5 retrieved postings first; they have analytics role-fit signals.
- Grounding caveat: Authorization language is reported only when it appears explicitly in retrieved source text. An advisor must verify the original posting; no eligibility is inferred.

Top retrieved evidence:

- `HF-734288` - Jr. Data Analyst (No OPT/CPT) at Winorbit Technology (Austin, TX), `high_fit`, score 6.862.
- `HF-250208` - Data Analyst (US REMOTE) at LeanTaaS (Anywhere), `high_fit`, score 6.858.
- `HF-697496` - Data Analyst ( US REMOTE) at Zip (New York, NY), `high_fit`, score 6.026.

## Thin Metadata

- Business use: Investigate an incomplete posting
- Predicted query label: `unclear`
- Headline: Review 5 retrieved postings first; they have analytics role-fit signals.
- Grounding caveat: Do not infer CPT, OPT, or sponsorship from this public dataset. Authorization evidence must be reviewed separately.

Top retrieved evidence:

- `HF-565767` - ESM Data Analytics Internship - Summer 2023 at TransUnion (Chicago, IL), `high_fit`, score 5.214.
- `HF-590956` - Data Analytics Internship – Monitoring of Telematic Devices... at Continental (Toulouse, France), `medium_fit`, score 5.267.
- `HF-726910` - Data Science & Analytics Internship - Summer 2024 at TransUnion (United States), `medium_fit`, score 5.152.
