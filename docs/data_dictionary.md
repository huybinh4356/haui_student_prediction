# Data Dictionary — HaUI Student Performance Prediction

## Identifier Fields (Not ML Features)

| Field | Type | Purpose |
|-------|------|---------|
| student_id | string | MSV; linking and deduplication only |
| full_name | string | Display only; privacy-protected |

## Demographic Fields

| Field | Type | Notes |
|-------|------|-------|
| date_of_birth | date | Optional; do not use alongside birth_year as two independent features |
| birth_year | integer | Alternative to date_of_birth |
| hometown | string | EDA aggregate only |
| major | string | EDA; evaluate sample size before ML use |

## Course Context

| Field | Type | Notes |
|-------|------|-------|
| course_id | string | Determines assessment profile |
| course_name | string | Display |
| semester | string | Temporal grouping |
| has_tx2 | boolean | Configuration; True if course has TX2 |

## Score Components

| Field | Type | Scale | ML Use |
|-------|------|-------|--------|
| tx1_score | float | 0-10 (to confirm) | Feature |
| tx2_score | float/null | 0-10 (to confirm) | Feature if has_tx2=True |
| midterm_score | float | 0-10 (to confirm) | Feature |
| target_final_exam_score | float | 0-10 (confirm from source) | Historical training target only; never in prediction input |

## Attendance & LMS

| Field | Type | Range | Notes |
|-------|------|-------|-------|
| absence_percentage | float | 0-100 | Source-provided; do not recalculate |
| lms_learning_percentage | float | 0-100 | Source-provided participation metric |

## Survey Fields

| Field | Type | Notes |
|-------|------|-------|
| interaction_score | float/ordinal | Higher = better interaction (to confirm) |
| stress_score | float/ordinal | Higher = more stress (to confirm) |
| motivation_score | float/ordinal | Higher = more motivation (to confirm) |
| works_part_time | boolean | Direction of effect unconfirmed; neutral mapping |
| survey_timestamp | datetime | Must precede prediction snapshot |

## System Fields

| Field | Type | Purpose |
|-------|------|---------|
| snapshot_date | datetime | Prediction cutoff; anti-leakage |
| source_file | string | Provenance |
| dataset_version | string | Reproducibility |
