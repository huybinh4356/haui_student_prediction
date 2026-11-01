import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_train_test(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe is None or len(dataframe) < 5:
        raise ValueError("Cần tối thiểu 5 bản ghi để chia tập train/test.")

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size phải nằm trong khoảng 0.1 đến 0.4.")

    test_rows = min(
        max(2, int(round(len(dataframe) * test_size))),
        len(dataframe) - 3,
    )
    if test_rows < 2:
        raise ValueError("Không đủ dữ liệu để tạo tập kiểm thử độc lập.")

    if "student_id" in dataframe.columns:
        identifiers = dataframe["student_id"]
        normalized_identifiers = identifiers.astype("string").str.strip().str.casefold()
        nonblank = identifiers.notna() & normalized_identifiers.ne("")
        valid_identifiers = normalized_identifiers.loc[nonblank]
        has_repeated_students = valid_identifiers.duplicated().any()

        if has_repeated_students:
            if not nonblank.all():
                raise ValueError(
                    "Dữ liệu có sinh viên lặp nhưng một số dòng thiếu student_id; "
                    "không thể chia train/test độc lập một cách an toàn."
                )

            if valid_identifiers.nunique() < 2:
                raise ValueError(
                    "Dữ liệu chỉ có một student_id lặp lại; "
                    "không thể tạo tập train/test độc lập theo sinh viên."
                )

            splitter = GroupShuffleSplit(
                n_splits=32,
                test_size=test_rows / len(dataframe),
                random_state=random_state,
            )
            selected_split = None
            selected_distance = None
            for train_indices, test_indices in splitter.split(
                dataframe,
                groups=normalized_identifiers.astype(str),
            ):
                if len(train_indices) < 3 or len(test_indices) < 2:
                    continue
                distance = abs(len(test_indices) - test_rows)
                if selected_distance is None or distance < selected_distance:
                    selected_split = (train_indices, test_indices)
                    selected_distance = distance

            if selected_split is None:
                raise ValueError(
                    "Không thể tạo tập train/test đủ lớn mà không tách cùng sinh viên "
                    "sang cả hai tập. Cần dữ liệu lịch sử đa dạng hơn."
                )

            train_indices, test_indices = selected_split
            train_data = dataframe.iloc[train_indices].copy()
            test_data = dataframe.iloc[test_indices].copy()

            train_students = set(normalized_identifiers.iloc[train_indices])
            test_students = set(normalized_identifiers.iloc[test_indices])
            if train_students & test_students:
                raise RuntimeError("Phát hiện student_id giao nhau giữa train và test.")

            return train_data, test_data

    train_data, test_data = train_test_split(
        dataframe,
        test_size=test_rows,
        random_state=random_state,
        shuffle=True,
    )
    if len(train_data) < 3 or len(test_data) < 2:
        raise ValueError("Tập train/test không đủ số bản ghi.")
    return train_data.copy(), test_data.copy()
