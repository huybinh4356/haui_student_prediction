# Risk Scoring Methodology

## Rule-Based Risk Engine (MVP)

The MVP uses a weighted rule-based system, not an ML classifier. Risk scores are for advisory purposes only.

## Hard Rule: Attendance Override

| absence_percentage | Status |
|-------------------:|--------|
| > 30% | `override_triggered = True`, status "REPEAT_REQUIRED" |
| 20-30% | High risk |
| 10-<20% | Medium risk |
| 5-<10% | Low risk |
| 0-<5% | Very low risk |

At exactly 30%, the attendance override does not trigger. Above 30%, the override is independent of other factors.

## Component Weights (Total 100%)

| Component | Weight | Direction |
|-----------|--------|-----------|
| Absence | 75% | Higher absence = higher risk |
| Academic score | 8% | Lower score = higher risk |
| LMS participation | 8% | Lower LMS = higher risk |
| Interaction | 4% | Lower interaction = higher risk |
| Work part-time | 2% | Neutral (unconfirmed direction; 50 risk points for either status) |
| Stress | 1.5% | Higher stress = higher risk (to confirm) |
| Motivation | 1.5% | Lower motivation = higher risk (to confirm) |

## LMS Risk Mapping (Inverse)

| lms_learning_percentage | Risk Level |
|------------------------:|------------|
| < 5% | Very high |
| 5-<10% | High |
| 10-<20% | Medium |
| 20-<30% | Low |
| 30-100% | Very low |

## Aggregate Risk Labels

| Score | Label |
|------:|-------|
| 0-<20 | Very low |
| 20-<40 | Low |
| 40-<60 | Medium |
| 60-<80 | High |
| 80-100 | Very high |

## Required inputs and validation

A risk result is evaluated only when all of the following values are present, finite, and within the configured range:

| Field | Allowed range |
|-------|---------------|
| `absence_percentage` | 0-100 |
| `lms_learning_percentage` | 0-100 |
| `predicted_course_score` | 0-10 |
| `interaction_score` | 1-5 |
| `stress_score` | 1-5 |
| `motivation_score` | 1-5 |

Missing columns, blank cells, non-numeric values, non-finite values, or out-of-range values cause that student's risk evaluation to be marked as not evaluated. The application displays the specific fields that need correction. It does not replace missing risk inputs with zero, averages, or other fabricated values. A missing risk input does not by itself cancel a valid final-exam score prediction.

## Important Notes

- Thresholds are prototype assumptions until calibrated against representative HaUI historical data.
- Missing or non-finite required inputs result in a clear error rather than a fabricated risk value.
- Part-time employment currently receives a neutral risk contribution because its direction is unconfirmed.
- This is not an official HaUI policy document.
