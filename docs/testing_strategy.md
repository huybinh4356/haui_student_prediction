# Testing Strategy

## Test Organization

```
tests/
├── unit/              # Fast, isolated tests
├── integration/       # Cross-module workflows
└── contracts/         # Input/output contracts
```

## Unit Test Coverage

| Module | Test File | Focus |
|--------|-----------|-------|
| data/ingestion | test_ingestion.py | File reading, validation, booleans, missing values, duplicates |
| data/cleaning | test_ingestion.py | Type conversion and empty-data handling |
| scoring/ | test_course_scoring.py | Formula correctness and TX2 handling |
| risk/ | test_risk_calculator.py, test_risk_service.py | Risk boundaries, weights and missing fields |
| features/ | test_ml_pipeline.py | Feature contract, feature builder and leakage checks |
| ml/ | test_ml_pipeline.py, test_xgboost_regressor.py | Preprocessing, baseline pipeline, XGBoost training, evaluation and edge cases |
| services/ | test_ml_pipeline.py, test_dataset_service.py | Prediction schema, missing-model behavior and failed persistence |
| storage/ | test_sqlite_repository.py | SQLite persistence, missing tables, invalid identifiers and replacement behavior |

## Running Tests

Run all unit tests:

```powershell
python -m pytest tests/unit -v
```

Run storage tests:

```powershell
python -m pytest tests/unit/test_sqlite_repository.py tests/unit/test_dataset_service.py -v
```

Run the XGBoost test:

```powershell
python -m pytest tests/unit/test_xgboost_regressor.py -v
```

## CI Checks

```powershell
ruff check src tests
python -m compileall src tests
python -m pytest tests/unit
```

## Test Data

- Tests use synthetic data.
- Real student data must not be committed.
- Fixtures are stored in `tests/fixtures/`.
- A written test is not evidence that it has passed; record the actual test run result.
