# Synthetic training dataset

This file contains 5,000 generated records and is not real HaUI student data.

- One row represents one synthetic student in one course context.
- Every student_id is unique in this file.
- target_final_exam_score is the historical-label stand-in used only for supervised training and evaluation. It must never be supplied as an inference feature.
- tx2_score is intentionally missing for a subset of records. Missing values are not replaced with zero.
- Scores use a simulated 0–10 scale; percentage fields use 0–100.
- The data is generated with seed 42 by scripts/generate_synthetic_training_data.py.
- The synthetic relationships are illustrative and do not establish real-world predictive accuracy or causal effects.

Regenerate with:

```powershell
python scripts/generate_synthetic_training_data.py --rows 5000 --seed 42
```

For an initial reproducible split, reserve 70% for training, 15% for validation, and 15% for the final test set. Do not tune model settings against the final test set.
