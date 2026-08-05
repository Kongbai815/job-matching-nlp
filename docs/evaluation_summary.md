# Evaluation Summary

## Primary Results

| System | Rows | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| M2 TF-IDF baseline | 20,000 | 0.7926 | 0.7850 |
| M3 true PEFT LoRA | 4,000 | 0.8745 | 0.8725 |
| Final word+character TF-IDF SVM | 4,000 | 0.9838 | 0.9837 |

On the same fixed 4,000-row subset, macro F1 increases from 0.7914 to 0.9837 (+0.1923). The final model also exceeds the 2,000-row LoRA experiment by 0.1112 macro-F1 points while being much smaller and easier to reproduce.

## Leakage-Controlled Checks

| Slice | Rows | Macro F1 |
| --- | ---: | ---: |
| Fixed set after removing train-text duplicates | 3,751 | 0.9830 |
| Full validation, novel text only | 18,672 | 0.9790 |
| Full validation, unseen companies only | 5,570 | 0.9802 |

The original random split contains 1,328 validation rows whose exact text appears in training. Deduplication and the two robustness slices show that duplicate leakage is not driving the result, although all scores still measure agreement with weak labels.

## Retrieval and Grounding

- Four workflow cases with retrieved evidence: 4 / 4
- Top five results for the US search case satisfy the US constraint: 5 / 5
- Explicit authorization restrictions correctly force review: 1 / 1
- Unsupported authorization claims in the casebook: 0

## Error Analysis

The final evaluation contains 65 errors across 4,000 balanced rows. Per-label F1 is 0.996 high-fit, 0.977 medium-fit, 0.992 low-fit, and 0.970 unclear. The remaining limitation is not model capacity: it is label validity. A pilot must collect advisor-reviewed labels before interpreting these numbers as real-world decision quality.
