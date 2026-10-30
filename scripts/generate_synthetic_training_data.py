import argparse
import csv
import math
from pathlib import Path


class RandomSource:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296

    def normal(self) -> float:
        u = max(self.random(), 1e-12)
        v = self.random()
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

    def beta_like(self) -> float:
        x = self.random() ** 1.4
        y = self.random() ** 0.65
        return x / (x + y)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def generate_dataset(rows: int = 5000, seed: int = 42) -> list[dict[str, object]]:
    if rows < 100:
        raise ValueError("Cần tối thiểu 100 sinh viên để tạo dữ liệu mô phỏng.")

    rng = RandomSource(seed)
    majors = ["Công nghệ thông tin", "Khoa học máy tính", "Hệ thống thông tin", "Kỹ thuật phần mềm"]
    courses = ["IT101", "IT102", "IT201", "IT202"]
    semesters = ["2025-1", "2025-2", "2026-1"]
    records = []

    for index in range(1, rows + 1):
        absence = clamp(rng.beta_like() * 100, 0, 65)
        lms = clamp(92 - absence * 1.15 + rng.normal() * 18, 0, 100)
        tx1 = clamp(5.8 + 0.035 * lms - 0.045 * absence + rng.normal() * 1.25, 0, 10)
        midterm = clamp(5.5 + 0.035 * lms - 0.05 * absence + 0.22 * (tx1 - 5) + rng.normal() * 1.35, 0, 10)
        tx2 = clamp(5.6 + 0.03 * lms - 0.04 * absence + 0.2 * (tx1 - 5) + rng.normal() * 1.3, 0, 10)
        interaction = clamp(3.2 + 0.018 * lms - 0.025 * absence + rng.normal() * 0.8, 1, 5)
        stress = clamp(2.8 + 0.018 * absence - 0.008 * lms + rng.normal() * 0.8, 1, 5)
        motivation = clamp(3 + 0.018 * lms - 0.012 * absence + rng.normal() * 0.75, 1, 5)
        works = rng.random() < 0.32
        if rng.random() < 0.18:
            tx2 = None
        work_penalty = rng.random() * 0.35 if works else 0.0
        final_score = clamp(
            0.24 * tx1
            + 0.29 * midterm
            + 0.18 * (tx1 if tx2 is None else tx2)
            + 0.13 * (lms / 10)
            + 0.08 * interaction
            + 0.08 * motivation
            - 0.025 * absence
            - 0.08 * stress
            - work_penalty
            + rng.normal() * 0.85,
            0,
            10,
        )
        records.append({
            "student_id": f"SYN{index:06d}",
            "major": majors[int(rng.random() * len(majors))],
            "course_id": courses[int(rng.random() * len(courses))],
            "semester": semesters[int(rng.random() * len(semesters))],
            "absence_percentage": round(absence, 2),
            "lms_learning_percentage": round(lms, 2),
            "tx1_score": round(tx1, 2),
            "tx2_score": "" if tx2 is None else round(tx2, 2),
            "midterm_score": round(midterm, 2),
            "interaction_score": round(interaction, 2),
            "stress_score": round(stress, 2),
            "motivation_score": round(motivation, 2),
            "works_part_time": works,
            "target_final_exam_score": round(final_score, 2),
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/sample/synthetic_training_data.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = generate_dataset(args.rows, args.seed)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated {len(records)} synthetic student records: {output}")


if __name__ == "__main__":
    main()
