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
| has_tx2 | nullable boolean | True/False when explicit course configuration or observed course records establish TX2 applicability; unknown when the score is missing and no configuration is available |

## Score Components

| Field | Type | Scale | ML Use |
|-------|------|-------|--------|
| tx1_score | float | 0-10 | Feature; values outside the configured range are rejected |
| tx2_score | float/null | 0-10 | Feature if has_tx2=True; values outside the configured range are rejected |
| midterm_score | float | 0-10 | Feature; values outside the configured range are rejected |
| target_final_exam_score | float | 0-10 | Historical training target only; never in prediction input; validated before training |

## Attendance & LMS

| Field | Type | Range | Notes |
|-------|------|-------|-------|
| absence_percentage | float | 0-100 | Source-provided; do not recalculate |
| lms_learning_percentage | float | 0-100 | Source-provided participation metric |

## Survey Fields

| Field | Type | Notes |
|-------|------|-------|
| interaction_score | float/ordinal | 1-5; higher = better interaction |
| stress_score | float/ordinal | 1-5; higher = more stress |
| motivation_score | float/ordinal | 1-5; higher = more motivation |
| works_part_time | boolean | Direction of effect unconfirmed; neutral mapping |
| survey_timestamp | datetime | Must precede prediction snapshot |

## System Fields

| Field | Type | Purpose |
|-------|------|---------|
| snapshot_date | datetime | Prediction cutoff; anti-leakage |
| source_file | string | Provenance |
| dataset_version | string | Reproducibility |

## Validation rules and risk inputs

- Percentages `absence_percentage` and `lms_learning_percentage` must be within 0-100.
- Academic scores `tx1_score`, `tx2_score`, `midterm_score`, and the historical target `target_final_exam_score` must be within 0-10.
- Survey fields `interaction_score`, `stress_score`, and `motivation_score` must be within 1-5.
- Risk evaluation requires absence percentage, LMS percentage, interaction, stress, motivation, and a calculated `predicted_course_score` within 0-10.
- Missing risk inputs do not block a valid grade prediction. The affected record is marked as not evaluated for risk and lists the missing fields. Values outside the configured ranges are rejected during import or historical training. The system never fills missing risk values with invented values.
- `works_part_time` remains optional in the risk engine and currently uses a neutral contribution until its effect is supported by evidence.

## UCI benchmark-only fields

These fields belong to the isolated UCI benchmark workflow and are not aliases for the HaUI production schema.

| Field | Source | Meaning | Scale |
|-------|--------|---------|-------|
| period1_grade | G1 | First-period course grade | 0–10 after normalization |
| period2_grade | G2 | Second-period course grade | 0–10 after normalization |
| weekly_study_time_level | studytime | Weekly study-time category | 1–4 |
| past_class_failures | failures | Previous class failures | Count |
| absence_count | absences | Number of absences | Count, not percentage |
| extra_school_support | schoolsup | Extra educational support | yes/no |
| family_education_support | famsup | Family educational support | yes/no |
| paid_subject_classes | paid | Extra paid classes in the subject | yes/no |
| higher_education_intent | higher | Wants to pursue higher education | yes/no |
| home_internet_access | internet | Internet access at home | yes/no |
| target_course_grade | G3 | Final course/year grade, not a separately identified exam score | 0–10 after normalization |

The benchmark deliberately excludes demographic and unrelated social fields. UCI grades G1, G2, and G3 are provided on a 0–20 scale and are divided by two. The original UCI absence count is retained as a count; it is not converted into a percentage without a denominator.
