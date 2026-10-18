import pytest
import os
from haui_student_prediction.storage.h2_database import start_jvm, get_connection, shutdown_jvm, init_schema
from haui_student_prediction.storage.dataset_repository import DatasetRepository


@pytest.fixture(scope="module")
def h2_conn():
    os.environ["HAUI_H2_URL"] = "jdbc:h2:mem:test_h2;DB_CLOSE_DELAY=-1"
    start_jvm()
    conn = get_connection()
    init_schema(conn)
    yield conn
    conn.close()
    shutdown_jvm()


def test_schema_created(h2_conn):
    meta = h2_conn.getMetaData()
    rs = meta.getTables(None, None, "DATASETS", None)
    assert rs.next()
    rs.close()


def test_insert_and_select(h2_conn):
    stmt = h2_conn.createStatement()
    stmt.execute("INSERT INTO datasets (session_id, source_file, row_count, validation_status) VALUES ('s1', 'f.csv', 10, 'OK')")
    h2_conn.commit()
    rs = stmt.executeQuery("SELECT * FROM datasets WHERE session_id = 's1'")
    assert rs.next()
    assert rs.getInt("row_count") == 10
    rs.close()
    stmt.close()


def test_dataset_repository(h2_conn):
    repo = DatasetRepository(__import__("haui_student_prediction.storage.h2_database", fromlist=[""]))
    repo.save_dataset_meta("s2", "test.xlsx", 5, "VALID")
    meta = repo.get_dataset_meta("s2")
    assert meta is not None
    assert meta["row_count"] == 5
    assert meta["validation_status"] == "VALID"


def test_rollback_on_error(h2_conn):
    stmt = h2_conn.createStatement()
    stmt.execute("DELETE FROM datasets WHERE session_id = 'rollback_test'")
    h2_conn.commit()
    try:
        h2_conn.setAutoCommit(False)
        stmt.execute("INSERT INTO datasets (session_id, source_file, row_count, validation_status) VALUES ('rollback_test', 'f.csv', 1, 'OK')")
        raise RuntimeError("forced")
    except RuntimeError:
        h2_conn.rollback()
    finally:
        h2_conn.setAutoCommit(True)
    rs = stmt.executeQuery("SELECT * FROM datasets WHERE session_id = 'rollback_test'")
    assert not rs.next()
    rs.close()
    stmt.close()
