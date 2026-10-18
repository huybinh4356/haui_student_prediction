CREATE TABLE IF NOT EXISTS datasets (
    session_id VARCHAR(64) PRIMARY KEY,
    source_file VARCHAR(512),
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    row_count INT,
    validation_status VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS student_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    student_id VARCHAR(32) NOT NULL,
    full_name VARCHAR(255),
    date_of_birth VARCHAR(32),
    birth_year INT,
    hometown VARCHAR(255),
    major VARCHAR(128),
    course_id VARCHAR(32),
    semester VARCHAR(32),
    absence_percentage DOUBLE,
    lms_learning_percentage DOUBLE,
    tx1_score DOUBLE,
    tx2_score DOUBLE,
    midterm_score DOUBLE,
    actual_final_exam_score DOUBLE,
    interaction_score DOUBLE,
    stress_score DOUBLE,
    motivation_score DOUBLE,
    works_part_time BOOLEAN,
    has_tx2 BOOLEAN,
    snapshot_date VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    student_id VARCHAR(32) NOT NULL,
    course_id VARCHAR(32),
    semester VARCHAR(32),
    model_version VARCHAR(32),
    predicted_final_exam_score DOUBLE,
    predicted_course_score DOUBLE,
    profile_id VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    student_id VARCHAR(32) NOT NULL,
    course_id VARCHAR(32),
    semester VARCHAR(32),
    absence_status VARCHAR(64),
    weighted_score DOUBLE,
    risk_category VARCHAR(32),
    is_override BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
