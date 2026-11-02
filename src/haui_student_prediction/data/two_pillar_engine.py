"""
Module: Two-Pillar Educational Knowledge Engine (Động cơ tri thức sư phạm 2 Trụ Cột)

Kết hợp hai trụ cột nền tảng:
1. Trụ cột 1: Năng lực học thuật cơ sở (TX1, TX2, Giữa kỳ).
2. Trụ cột 2: Tác động hành vi, tâm lý và ngoại cảnh (LMS, Chuyên cần, Stress, Động lực, Làm thêm, Tương tác).
3. Sai số ngẫu nhiên thực tế (Aleatoric noise): Mô phỏng các yếu tố ngoại cảnh phòng thi (hên/xui, ôn trúng tủ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_two_pillar_score(
    tx1: float,
    midterm: float,
    tx2: float | None = None,
    absence_pct: float = 0.0,
    lms_pct: float = 50.0,
    stress: float = 3.0,
    motivation: float = 3.0,
    interaction: float = 3.0,
    works_part_time: bool = False,
    noise: float = 0.0,
) -> float:
    """Tính điểm thi cuối kỳ theo tri thức 2 Trụ Cột chuẩn hóa trên thang 0-10."""
    # 1. Trụ cột 1: Năng lực học thuật nền tảng (0 - 10)
    if tx2 is not None and np.isfinite(tx2):
        academic_score = 0.30 * tx1 + 0.30 * tx2 + 0.40 * midterm
    else:
        academic_score = 0.40 * tx1 + 0.60 * midterm

    # 2. Trụ cột 2: Tác động hành vi và tâm lý (Chuẩn hóa 0 - 10)
    lms_comp = float(np.clip(lms_pct / 10.0, 0.0, 10.0))
    # Vắng học phạt: 0% -> 10 điểm, 30% -> 5 điểm, >45% -> 0 điểm
    absence_comp = float(np.clip(10.0 - (absence_pct / 3.0) * 1.5, 0.0, 10.0))
    motivation_comp = float(np.clip((motivation - 1.0) * 2.5, 0.0, 10.0))
    interaction_comp = float(np.clip((interaction - 1.0) * 2.5, 0.0, 10.0))
    stress_comp = float(np.clip((5.0 - stress) * 2.5, 0.0, 10.0))
    job_comp = 6.0 if works_part_time else 8.5

    behavioral_score = (
        0.30 * lms_comp
        + 0.30 * absence_comp
        + 0.15 * motivation_comp
        + 0.10 * interaction_comp
        + 0.10 * stress_comp
        + 0.05 * job_comp
    )

    # 3. Kết hợp 2 Trụ Cột: 70% Học thuật + 30% Hành vi + Nhiễu ngoại cảnh
    raw_score = 0.70 * academic_score + 0.30 * behavioral_score + noise
    return float(np.clip(raw_score, 0.5, 9.8))


def apply_two_pillar_calibration(
    df: pd.DataFrame,
    target_column_name: str = "Điểm thi cuối kỳ thực tế",
    noise_std: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Áp dụng nguyên lý 2 Trụ Cột lên toàn bộ tập dữ liệu để chuẩn hóa nhãn học tập."""
    res = df.copy()
    rng = np.random.default_rng(random_state)
    n = len(res)
    noises = rng.normal(0.0, noise_std, size=n)

    # Column mapping helper
    tx1_s = pd.to_numeric(res.get("Điểm TX1", res.get("tx1_score", 5.0)), errors="coerce").fillna(5.0)
    mid_s = pd.to_numeric(res.get("Điểm giữa kỳ", res.get("midterm_score", 5.0)), errors="coerce").fillna(5.0)
    tx2_col = "Điểm TX2" if "Điểm TX2" in res.columns else ("tx2_score" if "tx2_score" in res.columns else None)
    tx2_s = pd.to_numeric(res[tx2_col], errors="coerce") if tx2_col else pd.Series(np.nan, index=res.index)
    abs_s = pd.to_numeric(res.get("Tỷ lệ nghỉ học (%)", res.get("absence_percentage", 0.0)), errors="coerce").fillna(0.0)
    lms_s = pd.to_numeric(res.get("Tỷ lệ học LMS (%)", res.get("lms_learning_percentage", 50.0)), errors="coerce").fillna(50.0)
    str_s = pd.to_numeric(res.get("Mức độ stress", res.get("stress_score", 3.0)), errors="coerce").fillna(3.0)
    mot_s = pd.to_numeric(res.get("Động lực học tập", res.get("motivation_score", 3.0)), errors="coerce").fillna(3.0)
    int_s = pd.to_numeric(res.get("Mức độ tương tác", res.get("interaction_score", 3.0)), errors="coerce").fillna(3.0)
    
    wpt_col = "Đi làm thêm" if "Đi làm thêm" in res.columns else ("works_part_time" if "works_part_time" in res.columns else None)
    if wpt_col:
        wpt_s = res[wpt_col].astype(str).str.lower().isin(["có", "true", "1", "yes"])
    else:
        wpt_s = pd.Series(False, index=res.index)

    final_scores = []
    for i in range(n):
        tx2_val = tx2_s.iloc[i] if tx2_col and pd.notna(tx2_s.iloc[i]) else None
        s = compute_two_pillar_score(
            tx1=float(tx1_s.iloc[i]),
            midterm=float(mid_s.iloc[i]),
            tx2=float(tx2_val) if tx2_val is not None else None,
            absence_pct=float(abs_s.iloc[i]),
            lms_pct=float(lms_s.iloc[i]),
            stress=float(str_s.iloc[i]),
            motivation=float(mot_s.iloc[i]),
            interaction=float(int_s.iloc[i]),
            works_part_time=bool(wpt_s.iloc[i]),
            noise=float(noises[i]),
        )
        final_scores.append(round(s, 1))

    res[target_column_name] = final_scores
    return res
