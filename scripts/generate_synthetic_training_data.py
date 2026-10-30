import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("Cần tối thiểu 100 sinh viên để tạo dữ liệu mô phỏng.")

    rng = np.random.default_rng(seed)
    absence = np.clip(rng.beta(1.5, 5.5, rows) * 100, 0, 65)
    lms = np.clip(92 - absence * 1.15 + rng.normal(0, 18, rows), 0, 100)
    tx1 = np.clip(5.8 + 0.035 * lms - 0.045 * absence + rng.normal(0, 1.25, rows), 0, 10)
    midterm = np.clip(5.5 + 0.035 * lms - 0.05 * absence + 0.22 * (tx1 - 5) + rng.normal(0, 1.35, rows), 0, 10)
    tx2 = np.clip(5.6 + 0.03 * lms - 0.04 * absence + 0.2 * (tx1 - 5) + rng.normal(0, 1.3, rows), 0, 10)
    interaction = np.clip(3.2 + 0.018 * lms - 0.025 * absence + rng.normal(0, 0.8, rows), 1, 5)
    stress = np.clip(2.8 + 0.018 * absence - 0.008 * lms + rng.normal(0, 0.8, rows), 1, 5)
    motivation = np.clip(3.0 + 0.018 * lms - 0.012 * absence + rng.normal(0, 0.75, rows), 1, 5)
    works = rng.choice([True, False], size=rows, p=[0.32, 0.68])
    majors = rng.choice(["Công nghệ thông tin", "Khoa học máy tính", "Hệ thống thông tin", "Kỹ thuật phần mềm"], size=rows)
    course_ids = rng.choice(["IT101", "IT102", "IT201", "IT202"], size=rows)
    tx2_missing = rng.random(rows) < 0.18
    tx2[tx2_missing] = np.nan
    work_penalty = works.astype(float) * rng.uniform(0.0, 0.35, rows)
    final_score = (
        0.24 * tx1
        + 0.29 * midterm
        + 0.18 * np.nan_to_num(tx2, nan=tx1)
        + 0.13 * (lms / 10)
        + 0.08 * interaction
        + 0.08 * motivation
        - 0.025 * absence
        - 0.08 * stress
        - work_penalty
        + rng.normal(0, 0.85, rows)
    )
    final_score = np.clip(final_score, 0, 10)
    frame = pd.DataFrame({
        "student_id": [f"SYN{i:06d}" for i in range(1, rows + 1)],
        "major": majors,
        "course_id": course_ids,
        "semester": rng.choice(["2025-1", "2025-2", "2026-1"], size=rows),
        "absence_percentage": np.round(absence, 2),
        "lms_learning_percentage": np.round(lms, 2),
        "tx1_score": np.round(tx1, 2),
        "tx2_score": np.round(tx2, 2),
        "midterm_score": np.round(midterm, 2),
        "interaction_score": np.round(interaction, 2),
        "stress_score": np.round(stress, 2),
        "motivation_score": np.round(motivation, 2),
        "works_part_time": works,
        "target_final_exam_score": np.round(final_score, 2),
    })
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/sample/synthetic_training_data.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(args.rows, args.seed)
    dataset.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Generated {len(dataset)} synthetic student records: {output}")


if __name__ == "__main__":
    main()
