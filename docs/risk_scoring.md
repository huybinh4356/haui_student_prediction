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
| Work part-time | 2% | Interaction term: no extra risk when not working; when working, higher absence risk and lower LMS participation increase this component |
| Stress | 1.5% | Higher stress = higher risk (to confirm) |
| Motivation | 1.5% | Lower motivation = higher risk (to confirm) |

## Work and study interaction

Part-time work is not treated as a risk by itself. The work component represents the additional interaction between working while attendance and LMS participation are weak:

- If `works_part_time` is `False`, `work_risk_points = 0`.
- If `works_part_time` is `True`, `work_risk_points = (absence_risk_points + lms_risk_points) / 2`.
- The existing absence and LMS normalization rules are reused. No new LMS threshold is introduced.
- The work component remains weighted at 2%; all other component weights remain unchanged.
- This is an explicit interaction term: it adds a limited conditional contribution only for students who work part-time, without changing the separate absence or LMS calculations.

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
| `works_part_time` | boolean |

Missing columns or blank cells, including an unknown `works_part_time` value, cause that student's risk evaluation to be marked as not evaluated, while a valid final-exam score prediction may still be shown. The application displays the specific missing fields. Non-numeric, non-finite, or out-of-range source values are rejected during import or historical training; the risk service repeats its checks for records supplied through other paths. It does not replace missing risk inputs with zero, averages, or other fabricated values.

## Important Notes

- Thresholds are prototype assumptions until calibrated against representative HaUI historical data.
- Missing or non-finite required inputs result in a clear error rather than a fabricated risk value.
- Part-time employment does not add risk by itself; its 2% interaction contribution depends on the existing absence and LMS risk scores.
- This is not an official HaUI policy document.
