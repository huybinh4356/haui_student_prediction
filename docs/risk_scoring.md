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

At exactly 30%, override does NOT trigger. Above 30%, override is independent of other factors.

## Component Weights (Total 100%)

| Component | Weight | Direction |
|-----------|--------|-----------|
| Absence | 75% | Higher absence = higher risk |
| Academic score | 8% | Lower score = higher risk |
| LMS participation | 8% | Lower LMS = higher risk |
| Interaction | 4% | Lower interaction = higher risk |
| Work part-time | 2% | Neutral (unconfirmed direction) |
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

## Important Notes

- All thresholds are **prototype assumptions** pending calibration with HaUI historical data.
- Missing required fields result in `ValueError` with clear field name, not default risk assignment.
- `works_part_time` uses neutral mapping (50 risk points) until data confirms direction.
- This is NOT an official HaUI policy document.
