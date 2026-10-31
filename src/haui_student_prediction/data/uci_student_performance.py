from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd


UCI_DATASET_ID = 320
UCI_DATASET_NAME = "Student Performance"
UCI_DATASET_DOI = "10.24432/C5TG7T"
UCI_DATASET_LICENSE = "CC BY 4.0"
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/320/student%2Bperformance.zip"

SUBJECT_FILES = {
    "por": "student-por.csv",
    "mat": "student-mat.csv",
}

NUMERICAL_FEATURES = [
    "period1_grade",
    "period2_grade",
    "weekly_study_time_level",
    "past_class_failures",
    "absence_count",
]

CATEGORICAL_FEATURES = [
    "extra_school_support",
    "family_education_support",
    "paid_subject_classes",
    "higher_education_intent",
    "home_internet_access",
]

TARGET_COLUMN = "target_course_grade"

SOURCE_COLUMNS = {
    "G1": "period1_grade",
    "G2": "period2_grade",
    "studytime": "weekly_study_time_level",
    "failures": "past_class_failures",
    "absences": "absence_count",
    "schoolsup": "extra_school_support",
    "famsup": "family_education_support",
    "paid": "paid_subject_classes",
    "higher": "higher_education_intent",
    "internet": "home_internet_access",
    "G3": TARGET_COLUMN,
}

REQUIRED_SOURCE_COLUMNS = list(SOURCE_COLUMNS)


def _zip_bytes_have_required_files(archive_bytes: bytes) -> bool:
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            members = archive.namelist()
            available_files = {Path(name).name for name in members}
            expected_files = set(SUBJECT_FILES.values())
            if expected_files.issubset(available_files):
                return True

            nested_archives = [
                name for name in members
                if Path(name).name.lower() == "student.zip"
            ]
            for nested_name in nested_archives:
                try:
                    nested_bytes = archive.read(nested_name)
                    with zipfile.ZipFile(BytesIO(nested_bytes)) as nested:
                        nested_files = {
                            Path(name).name for name in nested.namelist()
                        }
                        if expected_files.issubset(nested_files):
                            return True
                except (KeyError, OSError, zipfile.BadZipFile):
                    continue
    except (OSError, zipfile.BadZipFile):
        return False
    return False


def _archive_has_required_files(archive_path: Path) -> bool:
    try:
        return _zip_bytes_have_required_files(archive_path.read_bytes())
    except OSError:
        return False


def _read_csv_from_archive(archive_path: Path, expected_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        direct_member = next(
            (name for name in members if Path(name).name == expected_name),
            None,
        )
        if direct_member is not None:
            with archive.open(direct_member) as stream:
                return pd.read_csv(stream, sep=";", encoding="utf-8")

        nested_name = next(
            (
                name for name in members
                if Path(name).name.lower() == "student.zip"
            ),
            None,
        )
        if nested_name is not None:
            nested_bytes = archive.read(nested_name)
            with zipfile.ZipFile(BytesIO(nested_bytes)) as nested:
                nested_member = next(
                    (
                        name for name in nested.namelist()
                        if Path(name).name == expected_name
                    ),
                    None,
                )
                if nested_member is not None:
                    with nested.open(nested_member) as stream:
                        return pd.read_csv(stream, sep=";", encoding="utf-8")

    raise FileNotFoundError(
        f"Không tìm thấy {expected_name} ở trong gói ZIP UCI, kể cả ZIP lồng bên trong."
    )


def _download_archive(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "student+performance.zip"
    manual_archive_path = raw_dir / "student.zip"

    if _archive_has_required_files(manual_archive_path):
        return manual_archive_path
    if _archive_has_required_files(archive_path):
        return archive_path
    archive_path.unlink(missing_ok=True)

    request = Request(
        UCI_DOWNLOAD_URL,
        headers={
            "User-Agent": "Mozilla/5.0 HaUI-Student-Prediction/1.0",
            "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            archive_bytes = response.read()
            content_type = response.headers.get("Content-Type", "không xác định")
            final_url = response.geturl()
    except Exception as exc:
        raise RuntimeError(
            "Không thể tải bộ dữ liệu UCI Student Performance. "
            "Kiểm tra kết nối mạng hoặc tải tệp student.zip từ trang UCI rồi đặt vào "
            f"{raw_dir}."
        ) from exc

    try:
        if not _zip_bytes_have_required_files(archive_bytes):
            raise ValueError(
                "Gói ZIP không chứa student-por.csv và student-mat.csv, "
                "kể cả khi các tệp CSV được đặt trong student.zip bên trong."
            )
    except (zipfile.BadZipFile, ValueError) as exc:
        raise RuntimeError(
            "Tệp tải về không phải bộ dữ liệu UCI Student Performance hợp lệ. "
            f"Máy chủ trả về Content-Type '{content_type}', URL cuối: {final_url}. "
            "Có thể mạng trường, proxy hoặc phần mềm bảo mật đã thay nội dung tải về. "
            "Hãy tải student.zip trực tiếp từ trang UCI và đặt vào thư mục "
            f"{raw_dir}, sau đó chạy lại. Ứng dụng sẽ tự đọc tệp này."
        ) from exc

    archive_path.write_bytes(archive_bytes)
    return archive_path


def _read_source(subject: str, data_file: str | Path | None, raw_dir: str | Path | None) -> pd.DataFrame:
    if data_file is not None:
        path = Path(data_file)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp UCI CSV: {path}")
        return pd.read_csv(path, sep=";", encoding="utf-8")

    base_dir = Path(raw_dir) if raw_dir is not None else Path("data/raw/uci_student_performance")
    archive_path = _download_archive(base_dir)
    expected_name = SUBJECT_FILES[subject]

    try:
        return _read_csv_from_archive(archive_path, expected_name)
    except (OSError, zipfile.BadZipFile, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Không thể đọc {expected_name} trong gói dữ liệu UCI đã tải."
        ) from exc


def load_uci_student_performance(
    subject: str = "por",
    data_file: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    normalized_subject = str(subject).strip().lower()
    if normalized_subject not in SUBJECT_FILES:
        raise ValueError("subject phải là 'por' (Portuguese) hoặc 'mat' (Mathematics).")

    raw = _read_source(normalized_subject, data_file, raw_dir)
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"UCI dataset thiếu các cột bắt buộc: {missing}")

    selected = raw[REQUIRED_SOURCE_COLUMNS].copy()

    numeric_source_columns = ["G1", "G2", "G3", "studytime", "failures", "absences"]
    for column in numeric_source_columns:
        original = selected[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Cột UCI '{column}' chứa giá trị không phải số.")
        selected[column] = converted

    if selected["G3"].isna().any():
        raise ValueError("UCI G3 có giá trị thiếu; không thể dùng làm nhãn huấn luyện.")

    for column in ["G1", "G2", "G3"]:
        values = selected[column].dropna()
        if ((values < 0) | (values > 20)).any():
            raise ValueError(f"Cột UCI '{column}' phải nằm trong thang điểm 0–20.")

    for column in ["studytime", "failures", "absences"]:
        values = selected[column].dropna()
        if (values < 0).any():
            raise ValueError(f"Cột UCI '{column}' không được chứa giá trị âm.")

    if not selected["studytime"].dropna().between(1, 4).all():
        raise ValueError("UCI studytime phải nằm trong các mức từ 1 đến 4.")

    result = selected.rename(columns=SOURCE_COLUMNS)
    for column in ["period1_grade", "period2_grade", TARGET_COLUMN]:
        result[column] = result[column] / 2.0

    for column in CATEGORICAL_FEATURES:
        values = result[column].astype("string").str.strip().str.lower()
        invalid = values.notna() & ~values.isin(["yes", "no"])
        if invalid.any():
            raise ValueError(f"Cột UCI '{column}' chỉ chấp nhận giá trị yes/no.")
        result[column] = values.astype(object)

    ordered_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    result = result[ordered_columns].copy()

    target_values = result[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise ValueError("UCI target G3 chứa giá trị không hữu hạn.")

    if len(result) < 20:
        raise ValueError("UCI dataset có ít hơn 20 bản ghi hợp lệ sau khi xử lý.")

    return result.reset_index(drop=True)
