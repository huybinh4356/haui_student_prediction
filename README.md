# HaUI Student Performance & Academic Risk Prediction System

Desktop application for analyzing student performance data, predicting final exam scores, and assessing academic risk for HaUI (Hanoi University of Industry).

## Requirements

- Python 3.10 or newer
- Windows 10/11 is the primary development environment
- XGBoost is used by the primary final-exam score prediction model

The session dataset store uses Python's built-in `sqlite3` module.

## Installation

Clone the repository:

```powershell
git clone https://github.com/huybinh4356/haui_student_prediction.git
cd haui_student_prediction
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the application and development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify that XGBoost is installed in the active Python environment:

```powershell
python -c "import xgboost; print(xgboost.__version__)"
```

Alternatively, install the listed dependencies first and then install the project in editable mode:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Windows, you can also run `setup.bat` to create or reuse `.venv`, install the project and development dependencies, and verify XGBoost. Start the application with `run.bat`.

## Run the application

From the repository root with the virtual environment activated:

```powershell
python main.py
```

The SQLite session database is stored at `data/session_student.db` by default. The application creates the parent directory when needed. Set `APP_BASE_DIR` to override the project data root when required.

## Data import

The application supports CSV and Excel (`.xls`, `.xlsx`) files. Common fields include:

- `student_id` (MSV)
- `absence_percentage` (0–100)
- `tx1_score`, `midterm_score`
- Optional: `tx2_score`, `lms_learning_percentage`, `interaction_score`, `stress_score`, `motivation_score`, `works_part_time`

Import workflow:

1. Select a file in the UI.
2. Map and validate columns.
3. Review errors, warnings, and duplicate rows.
4. Confirm the import.

An import is not reported as successful if writing the validated dataset to SQLite fails.

## Training model

When historical data with `actual_final_exam_score` is available:

1. Import the validated dataset.
2. Use the model training workflow in the UI or supported script.
3. The application trains the configured XGBoost regressor and evaluates it on a held-out test set.
4. Model artifacts are saved to `models/artifacts/`.
5. Registry metadata is saved to `models/registry.json`.

## Tests

Run the unit test suite:

```powershell
python -m pytest tests/unit -v
```

Run SQLite repository tests:

```powershell
python -m pytest tests/unit/test_sqlite_repository.py tests/unit/test_dataset_service.py -v
```

## Linting

```powershell
ruff check src tests
```

## Project structure

```text
haui_student_prediction/
├── main.py
├── pyproject.toml
├── requirements.txt
├── src/haui_student_prediction/
│   ├── data/
│   ├── features/
│   ├── ml/
│   ├── risk/
│   ├── scoring/
│   ├── services/
│   ├── storage/
│   ├── eda/
│   ├── ui/
│   └── config/
├── tests/
├── data/
├── models/
└── outputs/
```

## Known limitations

- Dataset persistence currently stores the latest imported dataset in the `current_session` table; a new successful import replaces that table.
- Model training requires historical data with `actual_final_exam_score`.
- Score scale and risk thresholds should be confirmed against the applicable HaUI grading policy and historical data.

Do not commit real student data or database files to version control.

## License

Internal academic use.
