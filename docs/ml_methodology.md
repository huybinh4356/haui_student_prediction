# ML Methodology — Final Exam Score Prediction

## Target

- `actual_final_exam_score`: historical final exam score
- Used only as target `y`, never as input feature

## Excluded from Features

- `student_id`, `full_name` (identifiers)
- `actual_final_exam_score` (target)
- Any post-exam computed scores

## Feature Candidates

- `tx1_score`, `tx2_score` (if available), `midterm_score`
- `absence_percentage`, `lms_learning_percentage`
- `interaction_score`, `stress_score`, `motivation_score`
- `works_part_time`, `major` (if sufficient samples)

## Pipeline

1. Validate schema & feature contract
2. Temporal/group split (when metadata supports)
3. Train/validation/test split
4. sklearn Pipeline: preprocessing → model
5. Preprocessing fit ONLY on training fold
6. Evaluation on held-out test set

## Models

- Baseline: `DummyRegressor` (mean/median)
- Primary: `LinearRegression`
- Candidates (when benchmarked): `Ridge`, `Lasso`, `RandomForestRegressor`, `HistGradientBoostingRegressor`

## Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (coefficient of determination)

## Anti-Leakage Measures

- Feature contract validates no target/ID in feature list
- Preprocessing within sklearn Pipeline, fit on train only
- No data from test fold influences training

## Model Registry

Each saved model includes:
- Model version
- Feature schema (numerical + categorical + target)
- Dataset fingerprint
- Training timestamp
- Library versions
- MAE/RMSE/R² on test set

## Limitations

- No model metrics without labeled historical data
- Model performance claims require validation on unseen data
- Do not present rule-based risk as ML prediction
