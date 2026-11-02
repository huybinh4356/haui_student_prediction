from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


@dataclass
class AccuracyEvaluationResult:
    # Core regression metrics
    total_samples: int
    mae: float
    rmse: float
    mse: float
    r2: float
    evs: float
    mape: float
    median_ae: float
    max_err: float
    pearson_corr: float
    spearman_corr: float

    # Distribution & spread
    actual_mean: float
    actual_std: float
    actual_min: float
    actual_max: float
    pred_mean: float
    pred_std: float
    pred_min: float
    pred_max: float
    spread_ratio: float
    unique_predictions: int
    clustering_warning: str

    # Baseline comparison
    baseline_mae: float
    baseline_rmse: float
    mae_improvement_pct: float
    rmse_improvement_pct: float
    relative_rmse: float
    beats_baseline: bool

    # Grade band accuracy
    grade_band_metrics: pd.DataFrame
    all_metrics_table: pd.DataFrame

    # Raw predictions and residuals
    predictions_df: pd.DataFrame


def compute_all_accuracy_metrics(
    actual: np.ndarray | pd.Series | list[float],
    predicted: np.ndarray | pd.Series | list[float],
    sample_ids: list[str] | None = None,
) -> AccuracyEvaluationResult:
    """Compute comprehensive accuracy attributes and metrics for regression models."""
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n_samples = len(y_true)

    if n_samples == 0:
        raise ValueError("Không có dữ liệu hợp lệ để tính toán độ chính xác.")

    # Core metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred)) if n_samples >= 2 else float("nan")
    evs = float(explained_variance_score(y_true, y_pred)) if n_samples >= 2 else float("nan")

    # Safe MAPE (avoid division by 0)
    nonzero_mask = np.abs(y_true) > 1e-4
    if np.any(nonzero_mask):
        mape = float(mean_absolute_percentage_error(y_true[nonzero_mask], y_pred[nonzero_mask])) * 100.0
    else:
        mape = float("nan")

    med_ae = float(median_absolute_error(y_true, y_pred))
    max_err = float(max_error(y_true, y_pred))

    # Correlations
    if n_samples >= 3 and np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        pearson_corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        # Spearman rank
        rank_true = pd.Series(y_true).rank()
        rank_pred = pd.Series(y_pred).rank()
        spearman_corr = float(np.corrcoef(rank_true, rank_pred)[0, 1])
    else:
        pearson_corr = float("nan")
        spearman_corr = float("nan")

    # Distribution statistics
    act_mean = float(np.mean(y_true))
    act_std = float(np.std(y_true))
    act_min = float(np.min(y_true))
    act_max = float(np.max(y_true))

    prd_mean = float(np.mean(y_pred))
    prd_std = float(np.std(y_pred))
    prd_min = float(np.min(y_pred))
    prd_max = float(np.max(y_pred))

    spread_ratio = float(prd_std / act_std) if act_std > 1e-12 else 0.0
    unique_preds = int(np.unique(np.round(y_pred, decimals=4)).size)

    # Clustering warning
    if spread_ratio < 0.30:
        clustering_warning = (
            f"CẢNH BÁO CO CỤM: Độ phân tán dự đoán chỉ bằng {spread_ratio:.1%} so với thực tế. "
            "Mô hình có xu hướng dự đoán co về điểm trung bình!"
        )
    elif unique_preds < min(10, n_samples // 3):
        clustering_warning = (
            f"CẢNH BÁO ĐỘ BIẾN THIÊN: Chỉ có {unique_preds} giá trị dự đoán khác biệt."
        )
    else:
        clustering_warning = "TỐT: Phổ dự đoán phân tán tự nhiên, phản ánh đúng năng lực học sinh."

    # Baseline comparison (predicting actual mean)
    baseline_pred = np.full(n_samples, act_mean)
    baseline_mae = float(mean_absolute_error(y_true, baseline_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))

    mae_imp = float((baseline_mae - mae) / baseline_mae * 100.0) if baseline_mae > 1e-12 else 0.0
    rmse_imp = float((baseline_rmse - rmse) / baseline_rmse * 100.0) if baseline_rmse > 1e-12 else 0.0
    relative_rmse = float(rmse / act_std) if act_std > 1e-12 else float("nan")
    beats_baseline = bool(mae < baseline_mae and r2 > 0.0)

    # Grade Band Metrics
    bands = [
        ("Giỏi / Xuất sắc (8.0 – 10.0)", y_true >= 8.0),
        ("Khá (6.5 – 7.9)", (y_true >= 6.5) & (y_true < 8.0)),
        ("Trung bình (5.0 – 6.4)", (y_true >= 5.0) & (y_true < 6.5)),
        ("Yếu / Nguy cơ (< 5.0)", y_true < 5.0),
    ]

    band_rows = []
    for band_name, band_mask in bands:
        count = int(np.sum(band_mask))
        if count > 0:
            b_mae = float(mean_absolute_error(y_true[band_mask], y_pred[band_mask]))
            b_rmse = float(np.sqrt(mean_squared_error(y_true[band_mask], y_pred[band_mask])))
            b_mean_diff = float(np.mean(y_pred[band_mask] - y_true[band_mask]))
        else:
            b_mae = float("nan")
            b_rmse = float("nan")
            b_mean_diff = float("nan")

        band_rows.append({
            "Khoảng điểm học lực": band_name,
            "Số lượng SV": count,
            "Tỷ lệ (%)": f"{count / n_samples * 100.0:.1f}%",
            "MAE (Sai số TB)": f"{b_mae:.3f}" if np.isfinite(b_mae) else "—",
            "RMSE": f"{b_rmse:.3f}" if np.isfinite(b_rmse) else "—",
            "Độ lệch trung bình (Bias)": f"{b_mean_diff:+.3f}" if np.isfinite(b_mean_diff) else "—",
        })
    band_df = pd.DataFrame(band_rows)

    # All metrics comprehensive table
    metrics_records = [
        {"Nhóm chỉ số": "Sai số tổng quát", "Thuộc tính độ chính xác": "MAE (Mean Absolute Error)", "Giá trị": f"{mae:.4f}", "Đơn vị": "Điểm", "Đánh giá": "Càng nhỏ càng tốt (độ lệch điểm trung bình)"},
        {"Nhóm chỉ số": "Sai số tổng quát", "Thuộc tính độ chính xác": "RMSE (Root Mean Squared Error)", "Giá trị": f"{rmse:.4f}", "Đơn vị": "Điểm", "Đánh giá": "Càng nhỏ càng tốt (phạt nặng sai số lớn)"},
        {"Nhóm chỉ số": "Sai số tổng quát", "Thuộc tính độ chính xác": "MSE (Mean Squared Error)", "Giá trị": f"{mse:.4f}", "Đơn vị": "Điểm²", "Đánh giá": "Phương sai sai số dự đoán"},
        {"Nhóm chỉ số": "Mức độ phù hợp", "Thuộc tính độ chính xác": "R² Score (Coefficient of Determination)", "Giá trị": f"{r2:.4f}", "Đơn vị": "Không thứ nguyên", "Đánh giá": f"Giải thích {r2 * 100:.1f}% biến thiên điểm số"},
        {"Nhóm chỉ số": "Mức độ phù hợp", "Thuộc tính độ chính xác": "Explained Variance Score (EVS)", "Giá trị": f"{evs:.4f}", "Đơn vị": "Không thứ nguyên", "Đánh giá": "Phương sai giải thích của mô hình"},
        {"Nhóm chỉ số": "Sai số phần trăm", "Thuộc tính độ chính xác": "MAPE (Mean Absolute % Error)", "Giá trị": f"{mape:.2f}%", "Đơn vị": "%", "Đánh giá": "Sai số phần trăm tuyệt đối trung bình"},
        {"Nhóm chỉ số": "Phân vị & Cực trị", "Thuộc tính độ chính xác": "Median Absolute Error", "Giá trị": f"{med_ae:.4f}", "Đơn vị": "Điểm", "Đánh giá": "Sai số trung vị (không bị ảnh hưởng bởi outlier)"},
        {"Nhóm chỉ số": "Phân vị & Cực trị", "Thuộc tính độ chính xác": "Max Prediction Error", "Giá trị": f"{max_err:.4f}", "Đơn vị": "Điểm", "Đánh giá": "Sai số lớn nhất trên toàn bộ tập dữ liệu"},
        {"Nhóm chỉ số": "Tương quan", "Thuộc tính độ chính xác": "Pearson Correlation (r)", "Giá trị": f"{pearson_corr:.4f}", "Đơn vị": "Không thứ nguyên", "Đánh giá": "Tương quan tuyến tính Thực tế vs Dự đoán"},
        {"Nhóm chỉ số": "Tương quan", "Thuộc tính độ chính xác": "Spearman Rank Correlation (ρ)", "Giá trị": f"{spearman_corr:.4f}", "Đơn vị": "Không thứ nguyên", "Đánh giá": "Tương quan thứ hạng năng lực sinh viên"},
        {"Nhóm chỉ số": "Độ phân tán", "Thuộc tính độ chính xác": "Prediction Spread Ratio", "Giá trị": f"{spread_ratio:.4f} ({spread_ratio * 100:.1f}%)", "Đơn vị": "Tỷ lệ", "Đánh giá": "> 50%: Tránh lỗi co cụm về trung bình"},
        {"Nhóm chỉ số": "Độ phân tán", "Thuộc tính độ chính xác": "Độ lệch chuẩn Dự đoán / Thực tế", "Giá trị": f"{prd_std:.3f} / {act_std:.3f}", "Đơn vị": "Điểm", "Đánh giá": "Độ biến thiên thực tế vs mô hình học được"},
        {"Nhóm chỉ số": "Độ phân tán", "Thuộc tính độ chính xác": "Khoảng điểm Thực tế vs Dự đoán", "Giá trị": f"[{act_min:.1f} – {act_max:.1f}] vs [{prd_min:.1f} – {prd_max:.1f}]", "Đơn vị": "Điểm", "Đánh giá": "Biên độ giá trị dự đoán"},
        {"Nhóm chỉ số": "So sánh Baseline", "Thuộc tính độ chính xác": "Baseline MAE (Dự đoán trung bình)", "Giá trị": f"{baseline_mae:.4f}", "Đơn vị": "Điểm", "Đánh giá": "Sai số khi chỉ đoán điểm trung bình"},
        {"Nhóm chỉ số": "So sánh Baseline", "Thuộc tính độ chính xác": "Mức cải thiện MAE so với Baseline", "Giá trị": f"{mae_imp:+.2f}%", "Đơn vị": "%", "Đánh giá": "Mức độ vượt trội so với mô hình tầm thường"},
        {"Nhóm chỉ số": "So sánh Baseline", "Thuộc tính độ chính xác": "Relative RMSE (RMSE / Std Actual)", "Giá trị": f"{relative_rmse:.4f}", "Đơn vị": "Tỷ lệ", "Đánh giá": "< 1.0 nghĩa là tốt hơn đoán trung bình"},
    ]
    all_metrics_df = pd.DataFrame(metrics_records)

    # Detailed predictions dataframe
    s_ids = sample_ids if sample_ids and len(sample_ids) == n_samples else [f"SV_{i+1:04d}" for i in range(n_samples)]
    pred_detail_df = pd.DataFrame({
        "sample_id": s_ids,
        "actual_score": y_true,
        "predicted_score": np.round(y_pred, 3),
        "residual": np.round(y_true - y_pred, 3),
        "absolute_error": np.round(np.abs(y_true - y_pred), 3),
    })

    return AccuracyEvaluationResult(
        total_samples=n_samples,
        mae=mae,
        rmse=rmse,
        mse=mse,
        r2=r2,
        evs=evs,
        mape=mape,
        median_ae=med_ae,
        max_err=max_err,
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        actual_mean=act_mean,
        actual_std=act_std,
        actual_min=act_min,
        actual_max=act_max,
        pred_mean=prd_mean,
        pred_std=prd_std,
        pred_min=prd_min,
        pred_max=prd_max,
        spread_ratio=spread_ratio,
        unique_predictions=unique_preds,
        clustering_warning=clustering_warning,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        mae_improvement_pct=mae_imp,
        rmse_improvement_pct=rmse_imp,
        relative_rmse=relative_rmse,
        beats_baseline=beats_baseline,
        grade_band_metrics=band_df,
        all_metrics_table=all_metrics_df,
        predictions_df=pred_detail_df,
    )

