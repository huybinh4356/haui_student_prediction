from pathlib import Path

import pandas as pd

from haui_student_prediction.storage.dataset_repository import DataRepository


def test_sqlite_database_created_and_data_persisted(tmp_path: Path):
    database_path = tmp_path / "session_student.db"
    repository = DataRepository(database_path)
    dataset = pd.DataFrame(
        {
            "student_id": ["S001", "S002"],
            "tx1_score": [7.0, 8.5],
        }
    )

    assert repository.save_session_data(dataset)
    assert database_path.exists()
    pd.testing.assert_frame_equal(repository.get_session_data(), dataset)
