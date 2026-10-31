# HaUI Student Performance & Academic Risk Prediction System

Desktop application for analyzing student performance data, predicting final exam scores, and assessing academic risk for HaUI (Hanoi University of Industry).

## Requirements

- Python 3.10 or newer
- Windows 10/11 is the primary development environment
- XGBoost is used by the primary HaUI final-exam score prediction model
- XGBoost, Random Forest, and CatBoost are used by the separate UCI benchmark comparison

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

Verify that the ML libraries are installed in the active Python environment:

```powershell
python -c "import xgboost, catboost; print('XGBoost:', xgboost.__version__); print('CatBoost:', catboost.__version__)"
```

Alternatively, install the listed dependencies first and then install the project in editable mode:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Windows, you can also run `setup.bat` to create or reuse `.venv`, install the project and development dependencies, and verify XGBoost and CatBoost. Start the application with `run.bat`.

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
2. Recognized Vietnamese and English column aliases are mapped to canonical fields, then validated.
3. Review errors, warnings, duplicate rows, and any incomplete duplicate checks.
4. Confirm the import.

Duplicate detection uses `student_id`, `course_id`, and `semester` together. If a key column or a row's key value is missing, the application warns that duplicate detection is incomplete for those records instead of treating missing key values as a confirmed duplicate.

An import is not reported as successful if writing the validated dataset to SQLite fails.

## Training model

Prediction data and model-training data are separate.

1. Import the current student dataset without any final-exam score column.
2. Open **Models** and select **Train XGBoost**.
3. Choose a separate historical CSV/Excel file containing a final-exam label. The loader converts that historical label to `target_final_exam_score`; it is used only as the supervised-learning target.
4. The XGBoost pipeline learns from available pre-exam features such as TX1/TX2, midterm, absence, LMS participation, interaction, stress, motivation, work status, major, and course ID when supported by the data.
5. The model is evaluated on a held-out test set with MAE, RMSE, and R², then registered if training completes successfully.
6. For current students, the system predicts the final-exam score, calculates the expected course total using the configured assessment profile, and evaluates academic risk.

The historical label must never be used as an input feature or included in the current prediction dataset. The amount and diversity of real historical data affect model reliability.

### UCI benchmark training (separate experimental workflow)

The repository includes an experimental benchmark based on the UCI Student Performance dataset (dataset ID 320). By default, it uses the Portuguese-subject file (student-por.csv, 649 records); Mathematics (student-mat.csv, 395 records) can be selected separately. The two subject files are not concatenated.

Run from the repository root:

    python scripts/train_uci_student_performance.py --subject por

The script downloads and caches the official UCI ZIP archive under data/raw/uci_student_performance/ when the data file is not supplied. That directory is ignored by Git. To use an already downloaded file:

    python scripts/train_uci_student_performance.py --subject por --data-file data/raw/uci_student_performance/student-por.csv

The benchmark keeps only these source attributes: G1, G2, studytime, failures, absences, schoolsup, famsup, paid, higher, and internet. Grades G1, G2, and G3 are divided by two to convert the original 0–20 scale to 0–10. The absence field remains a count of absences; it is not converted into an unsupported percentage. Demographic and unrelated social fields are excluded.

The desktop application includes a Vietnamese page named “Huấn luyện và so sánh” for running the UCI experiment and viewing its results without using the command line. The page uses the same benchmark workflow described below and does not replace the HaUI production model.

The data is split once into 80% Training and 20% held-out Testing. Five-fold cross-validation runs only inside the 80% Training partition: in each fold, four parts are used to fit the pipeline and the remaining part acts as Validation. This provides validation metrics without reserving a separate fixed validation set. XGBoost, Random Forest, and CatBoost use the same folds and the same untouched Test rows. The best model is ranked by mean cross-validation MAE; the Test metrics are reported only for final evaluation. Preprocessing is fitted independently inside each fold to prevent information leakage. The script writes cross-validation mean/standard deviation, holdout metrics, and row-level test predictions to outputs/uci_model_comparison/.

Important target limitation: UCI defines G3 as the final course/year grade, not as a separately identified final-exam score. The UCI benchmark therefore predicts target_course_grade; it is not registered as the production HaUI final-exam model and must not be presented as a model trained on HaUI exam scores. The production training workflow still requires appropriately labeled historical final-exam data.

Dataset attribution: Cortez, P. (2008), Student Performance, UCI Machine Learning Repository, DOI https://doi.org/10.24432/C5TG7T, CC BY 4.0.

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
- Model training requires a separate historical dataset with a target label; the current prediction dataset must not contain that label.
- Score scale and risk thresholds should be confirmed against the applicable HaUI grading policy and historical data.

Do not commit real student data or database files to version control.

## License

Internal academic use.
