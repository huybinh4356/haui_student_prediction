import numpy as np
import pandas as pd

from haui_student_prediction.data.column_mapping import is_training_target_column
from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.model_registry import ACTIVE_MODEL_VERSION, ModelRegistry
from haui_student_prediction.scoring.final_course_score import ScoringService


PREDICTION_FIELD_RANGES = {
    "absence_percentage": (0.0, 100.0),
    "lms_learning_percentage": (0.0, 100.0),
    "tx1_score": (0.0, 10.0),
    "tx2_score": (0.0, 10.0),
    "midterm_score": (0.0, 10.0),
    "interaction_score": (1.0, 5.0),
    "stress_score": (1.0, 5.0),
    "motivation_score": (1.0, 5.0),
}

REQUIRED_SCORE_FIELDS = ("tx1_score", "midterm_score")


class PredictionService:
    """Load the registered model, validate prediction inputs and calculate scores."""

    def __init__(self, registry: ModelRegistry, scoring_service: ScoringService):
        self.registry = registry
        self.scoring_service = scoring_service

    def is_model_ready(self, model_version: str = ACTIVE_MODEL_VERSION) -> bool:
        return self.registry.has_model(model_version)

    @staticmethod
    def _validate_numeric_inputs(
        records: pd.DataFrame,
        required_fields: set[str],
    ) -> None:
        missing_columns = sorted(required_fields - set(records.columns))
        if missing_columns:
            raise ValueError(
                "Thiếu cột điểm bắt buộc để dự đoán/tính điểm: "
                + ", ".join(missing_columns)
            )

        errors = []
        for column, (minimum, maximum) in PREDICTION_FIELD_RANGES.items():
            if column not in records.columns:
                continue

            raw = records[column]
            numeric = pd.to_numeric(raw, errors="coerce")
            blank = raw.astype("string").str.strip().eq("").fillna(False)
            present = raw.notna() & ~blank
            finite = pd.Series(
                np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
                index=records.index,
            )
            invalid = present & (~numeric.notna() | ~finite)
            invalid |= numeric.notna() & (
                (numeric < minimum) | (numeric > maximum)
            )
            if column in required_fields:
                invalid |= ~numeric.notna() | ~finite

            if invalid.any():
                rows = [str(index) for index in records.index[invalid][:8]]
                errors.append(
                    f"{column}: dòng {', '.join(rows)}; yêu cầu "
                    f"{minimum:g}–{maximum:g}"
                    + (" và không được để trống" if column in required_fields else "")
                )

        if errors:
            raise ValueError(
                "Dữ liệu đầu vào không hợp lệ. " + " | ".join(errors)
            )

    @staticmethod
    def _build_input_warnings(
        records: pd.DataFrame,
        contract: FeatureContract,
        required_fields: set[str],
    ) -> pd.Series:
        warnings_by_row = {index: [] for index in records.index}

        for column in contract.get_all_features():
            if column in required_fields:
                continue
            if column not in records.columns:
                for row_warnings in warnings_by_row.values():
                    row_warnings.append(
                        f"Thiếu cột {column}; giá trị được xử lý theo thống kê tập huấn luyện"
                    )
                continue

            values = records[column]
            missing = values.isna()
            if values.dtype == object or str(values.dtype).startswith("string"):
                missing |= values.astype("string").str.strip().eq("").fillna(False)

            for index in records.index[missing]:
                warnings_by_row[index].append(
                    f"Thiếu giá trị {column}; giá trị được xử lý theo thống kê tập huấn luyện"
                )

        return pd.Series(
            {
                index: (
                    "; ".join(row_warnings)
                    if row_warnings
                    else "Đầy đủ các trường đặc trưng đã khai báo"
                )
                for index, row_warnings in warnings_by_row.items()
            },
            index=records.index,
            dtype="object",
        )

    @staticmethod
    def _quality_status(metrics: dict) -> str:
        if not isinstance(metrics, dict):
            return "Chưa có đủ chỉ số đánh giá mô hình"
        try:
            model_mae = float(metrics.get("mae", float("nan")))
            baseline_mae = float(metrics.get("baseline_mae", float("nan")))
            spread_ratio = float(metrics.get("prediction_spread_ratio", float("nan")))
        except (TypeError, ValueError):
            return "Chưa có đủ chỉ số đánh giá mô hình"

        notes = []
        if np.isfinite(model_mae) and np.isfinite(baseline_mae):
            if model_mae > baseline_mae:
                notes.append("MAE chưa tốt hơn baseline dự đoán trung bình")
            else:
                notes.append("MAE tốt hơn hoặc bằng baseline dự đoán trung bình")
        else:
            notes.append("Chưa có MAE/baseline hợp lệ")

        if np.isfinite(spread_ratio) and spread_ratio < 0.5:
            notes.append("độ phân tán dự đoán thấp hơn nhiều so với điểm thực tế")
        return "; ".join(notes)

    def predict_student_results(
        self,
        student_records: pd.DataFrame,
        model_version: str,
        course_profile_id: str,
    ) -> pd.DataFrame:
        if not self.is_model_ready(model_version):
            raise RuntimeError(
                f"Model version '{model_version}' is not ready. "
                "No model artifact has been trained or registered. "
                "Train a model before requesting predictions."
            )

        if student_records is None or len(student_records) == 0:
            raise ValueError("Student records DataFrame is empty")

        profile = self.scoring_service.profiles.get(course_profile_id)
        if profile is None:
            raise ValueError("Invalid profile_id")

        excluded_columns = [
            column for column in student_records.columns
            if is_training_target_column(column)
        ]
        prediction_input = student_records.drop(
            columns=excluded_columns,
            errors="ignore",
        ).copy()

        required_score_fields = set(REQUIRED_SCORE_FIELDS)
        if profile.has_tx2:
            if "tx2_score" not in prediction_input.columns:
                raise ValueError(
                    "The selected course profile requires TX2, "
                    "but the input has no tx2_score column."
                )
            required_score_fields.add("tx2_score")
        elif "tx2_score" in prediction_input.columns:
            tx2_values = pd.to_numeric(
                prediction_input["tx2_score"],
                errors="coerce",
            )
            if tx2_values.notna().any():
                raise ValueError(
                    "The input contains TX2 scores, but the selected profile is for "
                    "courses without TX2. Select the profile with TX2 or separate "
                    "the dataset by course."
                )

        self._validate_numeric_inputs(prediction_input, required_score_fields)

        model, metadata = self.registry.load_model(model_version)
        features_meta = metadata.get("features")
        if not features_meta:
            raise ValueError(
                f"Registered model '{model_version}' has no feature metadata; "
                "cannot determine prediction schema"
            )

        contract = FeatureContract(**features_meta)
        contract_features = set(contract.get_all_features())
        model_required_fields = {"tx1_score", "midterm_score"}
        if profile.has_tx2:
            model_required_fields.add("tx2_score")
        missing_model_required = sorted(model_required_fields - contract_features)
        if missing_model_required:
            raise ValueError(
                "Mô hình hiện tại chưa được huấn luyện với các đặc trưng điểm bắt buộc: "
                + ", ".join(missing_model_required)
                + ". Hãy huấn luyện lại bằng dữ liệu lịch sử đầy đủ."
            )

        builder = FeatureBuilder(contract)
        input_warnings = self._build_input_warnings(
            prediction_input,
            contract,
            required_score_fields,
        )
        X_infer = builder.extract_features(prediction_input, is_training=False)

        raw_predictions = model.predict(X_infer)
        if isinstance(raw_predictions, tuple):
            raw_predictions = raw_predictions[0]
        predictions = np.asarray(raw_predictions, dtype=float).reshape(-1)

        if len(predictions) != len(prediction_input):
            raise RuntimeError(
                "Số lượng kết quả dự đoán không khớp với số bản ghi đầu vào."
            )

        finite_predictions = np.isfinite(predictions)
        in_range_predictions = (predictions >= 0.0) & (predictions <= 10.0)
        invalid_predictions = ~(finite_predictions & in_range_predictions)
        if invalid_predictions.any():
            rows = [
                str(index)
                for index in prediction_input.index[invalid_predictions][:8]
            ]
            raise RuntimeError(
                "Mô hình trả về điểm dự đoán không hợp lệ (phải hữu hạn và nằm "
                f"trong thang 0–10). Kiểm tra dòng: {', '.join(rows)}. "
                "Kết quả không bị tự động cắt ngưỡng."
            )

        results_df = prediction_input.copy()
        results_df["predicted_final_exam_score"] = predictions
        results_df["prediction_input_warnings"] = input_warnings

        metric_metadata = metadata.get("metrics", {})
        if not isinstance(metric_metadata, dict):
            metric_metadata = {}
        model_mae = metric_metadata.get("mae", float("nan"))
        cv_mae = metric_metadata.get("cv_mae_mean", float("nan"))
        baseline_mae = metric_metadata.get("baseline_mae", float("nan"))
        spread_ratio = metric_metadata.get("prediction_spread_ratio", float("nan"))

        results_df["model_holdout_mae_reference"] = model_mae
        results_df["model_cv_mae_reference"] = cv_mae
        results_df["model_baseline_mae_reference"] = baseline_mae
        results_df["model_prediction_spread_ratio"] = spread_ratio
        results_df["model_selected_candidate"] = metric_metadata.get(
            "selected_candidate",
            "unknown",
        )
        results_df["model_quality_status"] = self._quality_status(metric_metadata)

        course_totals = []
        course_score_status = []
        for _, row in results_df.iterrows():
            try:
                tx2 = row.get("tx2_score")
                tx2_value = float(tx2) if pd.notna(tx2) else None
                course_total = self.scoring_service.calculate_score(
                    profile_id=course_profile_id,
                    tx1=row["tx1_score"],
                    midterm=row["midterm_score"],
                    predicted_final=row["predicted_final_exam_score"],
                    tx2=tx2_value,
                )
                course_totals.append(course_total)
                course_score_status.append("Đã tính")
            except (KeyError, TypeError, ValueError) as exc:
                course_totals.append(np.nan)
                course_score_status.append(f"Chưa tính được điểm học phần: {exc}")

        results_df["predicted_course_score"] = course_totals
        results_df["course_score_status"] = course_score_status
        return results_df
