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
| data/ingestion | test_ingestion.py | File read, validation, boolean, missing, duplicates |
| data/cleaning | (via ingestion) | Type conversion, empty handling |
| scoring/ | test_course_scoring.py | Formula correctness, TX2 handling |
| risk/ | test_risk_calculator.py + test_risk_service.py | Boundaries, weights, missing data |
| features/ | test_ml_pipeline.py | Contract validation, builder, leakage |
| ml/ | test_ml_pipeline.py | Pipeline, train/eval, registry, edge cases |
| services/ | test_ml_pipeline.py | Prediction, schema mismatch, no-model |
| storage/ | test_h2_integration.py | H2 CRUD, rollback, schema |

## Running Tests

```bash
# All unit tests
python -m pytest tests/unit/ -v

# With coverage
python -m pytest tests/unit/ --cov=src --cov-report=html

# H2 tests (requires JDK)
python -m pytest tests/unit/test_h2_integration.py -v
```

## CI Checks

```bash
ruff check src tests
python -m compileall src tests
python -m pytest tests/unit/
```

## Test Data

- All tests use synthetic data
- No real student data in repository
- Fixtures in `tests/fixtures/`
