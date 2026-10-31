# ML Methodology — Final Exam Score Prediction

## Objective

Predict the final-exam score before the exam takes place. Use only information available at the prediction cutoff.

## Historical training target

- `target_final_exam_score` is a target label used only in the separate historical training dataset.
- The current student dataset used for prediction must not contain this label.
- The target label is never a feature and must never be copied into prediction results.

## Candidate features

- `tx1_score`, `tx2_score` when available, and `midterm_score`
- `absence_percentage` and `lms_learning_percentage`
- `interaction_score`, `stress_score`, and `motivation_score`
- `works_part_time`
- `major` and `course_id` when they vary and the training data supports them

Identifiers such as `student_id` and `full_name` are for display/grouping only, never model features. Date of birth, hometown, and other fields without a clear academic rationale are not automatically included.

## Model and evaluation

XGBoost regression is the primary model in the application. Preprocessing stays inside the scikit-learn Pipeline and is fitted on training data only. Evaluation uses a held-out test set and reports MAE, RMSE, and R² when defined. These metrics are not reliable when the historical dataset is too small or unrepresentative.

When historical records repeat the same `student_id`, train/test splitting groups records by normalized student ID so the same student cannot appear in both partitions. If there are not enough independent student groups, training stops with a clear error rather than falling back to a row-level split.

The model learns feature relationships from historical examples. Manual feature percentages should not be imposed on XGBoost. The final-course score is a separate deterministic calculation using the course assessment profile.

## End-to-end outputs

1. Predicted final-exam score from XGBoost
2. Predicted course total from the configured assessment weights
3. Academic-risk result from the risk rules, including the separate attendance override

## Leakage controls

- Historical target data is loaded in a separate training workflow.
- The prediction import removes any final-exam target column.
- The feature contract excludes the target and identifiers.
- The preprocessing pipeline is fitted only on the training partition.
- Held-out test rows are used for evaluation before the final registered model is fitted on the full labeled historical dataset.

## Model registry

Each registered model stores its version, dataset fingerprint, metrics, feature contract, and artifact filename. After the prediction schema or target contract changes, the active model version is bumped so older artifacts are not silently reused.

## Limitations

- A supervised regressor requires historical labeled examples.
- Synthetic fixtures are for software tests, not evidence of real-world prediction accuracy.
- Risk scores are rule-based indicators, not calibrated probabilities.
- Performance must be re-evaluated using representative historical data.

## UCI Student Performance benchmark (experimental)

A separate experimental workflow uses UCI Student Performance (dataset ID 320). The default subject is Portuguese (student-por.csv, 649 rows); Mathematics (student-mat.csv, 395 rows) can be trained independently. The two subject files are not concatenated because the source does not provide a stable student identifier suitable for grouping across subjects.

Only the following source attributes are selected: G1, G2, studytime, failures, absences, schoolsup, famsup, paid, higher, and internet. G1 and G2 are mapped to period-grade features and all three grade columns are divided by two to fit a 0–10 scale. Absences remains an absence count and is never mislabeled as an absence percentage. Demographic and unrelated social attributes are excluded.

The benchmark compares XGBoost, Random Forest, and CatBoost on the same held-out test rows. Each model has its own preprocessing pipeline; imputers, encoders, and scalers are fitted only on the training partition. The process produces aggregate MAE, RMSE, R², training time, and row-level predictions for direct comparison.

UCI defines G3 as the final course/year grade, while G1 and G2 are the first- and second-period grades. G3 is not a separately identified final-exam score. The UCI benchmark target is therefore named target_course_grade. These models are experimental, are not saved to the active HaUI production model registry, and cannot establish HaUI final-exam prediction accuracy.

Source: Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. License: CC BY 4.0.
