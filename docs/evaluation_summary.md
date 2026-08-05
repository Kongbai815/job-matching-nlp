# Evaluation Summary

## Primary Results

| System | Rows | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| M2 TF-IDF baseline | 20,000 | 0.7926 | 0.7850 |
| M3 true PEFT LoRA | 4,000 | 0.8745 | 0.8725 |
| Final end-to-end system | 4,000 | 0.8003 | 0.8006 |

The cleanest preliminary improvement claim is the same-subset comparison used in Milestone 4: macro F1 increases from 0.7914 to 0.8006 (+0.0092). Comparisons across M2, M3, and the final system are directional because the systems and evaluation sets are not identical.

## Retrieval and Grounding

- Retrieval success: 100.0%
- Support hit@5: 90.0%
- Four workflow cases with retrieved evidence: 4 / 4
- Unsupported authorization claims in the casebook: 0

## Error Analysis

The final evaluation contains 798 errors across 4,000 balanced rows. High-fit F1 is 0.960; medium-fit F1 is 0.725. The dominant errors are `medium_fit -> low_fit` (380) and `unclear -> low_fit` (254). These errors argue for a review queue, not automatic rejection.
