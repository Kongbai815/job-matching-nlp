# Architecture

The final system uses a retrieval-plus-grounded-output architecture.

- `msba_job_matcher.core.JobMatchingSystem` loads the 100k project CSV.
- It builds an inverted index over the 80k train postings.
- For a query, it retrieves similar postings using weighted sparse terms.
- A transparent rubric predicts the query-level fit label.
- The generator is a deterministic template that cites only retrieved source fields.

This design prioritizes reproducibility and governance over black-box automation.
