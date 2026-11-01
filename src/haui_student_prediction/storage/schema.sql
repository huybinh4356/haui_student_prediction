CREATE TABLE IF NOT EXISTS datasets (
    session_id TEXT PRIMARY KEY,
    source_file TEXT,
    imported_at TEXT DEFAULT (datetime('now')),
    row_count INTEGER,
    validation_status TEXT
);

CREATE TABLE IF NOT EXISTS student_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    full_name TEXT,
    date_of_birth TEXT,
    birth_year INTEGER,
    hometown TEXT,
    major TEXT,
    course_id TEXT,
    semester TEXT,
    absence_percentage REAL,
    lms_learning_percentage REAL,
    tx1_score REAL,
    tx2_score REAL,
    midterm_score REAL,
    interaction_score REAL,
    stress_score REAL,
    motivation_score REAL,
    works_part_time INTEGER,
    has_tx2 INTEGER,
    snapshot_date TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    course_id TEXT,
    semester TEXT,
    model_version TEXT,
    predicted_final_exam_score REAL,
    predicted_course_score REAL,
    profile_id TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS risk_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    course_id TEXT,
    semester TEXT,
    absence_status TEXT,
    weighted_score REAL,
    risk_category TEXT,
    is_override INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
