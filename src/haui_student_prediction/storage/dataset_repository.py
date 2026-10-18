import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DatasetRepository:
    def __init__(self, h2_database_module):
        self._db = h2_database_module

    def save_dataset_meta(self, session_id: str, source_file: str, row_count: int, validation_status: str) -> None:
        conn = self._db.get_connection()
        try:
            ps = conn.prepareStatement(
                "MERGE INTO datasets (session_id, source_file, row_count, validation_status) "
                "KEY(session_id) VALUES (?, ?, ?, ?)"
            )
            ps.setString(1, session_id)
            ps.setString(2, source_file)
            ps.setInt(3, row_count)
            ps.setString(4, validation_status)
            ps.executeUpdate()
            ps.close()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_dataset_meta(self, session_id: str) -> Optional[dict]:
        conn = self._db.get_connection()
        try:
            ps = conn.prepareStatement(
                "SELECT session_id, source_file, imported_at, row_count, validation_status "
                "FROM datasets WHERE session_id = ?"
            )
            ps.setString(1, session_id)
            rs = ps.executeQuery()
            if rs.next():
                return {
                    "session_id": rs.getString("session_id"),
                    "source_file": rs.getString("source_file"),
                    "imported_at": rs.getString("imported_at"),
                    "row_count": rs.getInt("row_count"),
                    "validation_status": rs.getString("validation_status"),
                }
            return None
        finally:
            conn.close()
