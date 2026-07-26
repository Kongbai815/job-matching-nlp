# Evaluation Summary

Baseline and final metrics use weak labels created from transparent MSBA advisor-screening rules.

- Milestone 2 TF-IDF baseline: accuracy 0.793, macro F1 0.785.
- Milestone 3 true PEFT LoRA: accuracy 0.875, macro F1 0.873.
- Final end-to-end system: accuracy 0.800, macro F1 0.801.

The final system adds grounded evidence and risk controls. The PEFT LoRA classifier remains the best adaptation experiment for pure label prediction, while the deployed prototype is optimized for advisor-facing evidence, grounding, and safe escalation. It should be evaluated next on human-reviewed labels and on a dataset containing authorization language.
