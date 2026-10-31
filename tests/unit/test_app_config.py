import os
from pathlib import Path

from haui_student_prediction import app_config


def test_load_config_resolves_app_base_dir_from_project_dotenv(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    configured_root = tmp_path / "configured_data_root"
    configured_root.mkdir()
    (project_root / ".env").write_text(
        f"APP_BASE_DIR={configured_root}\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("APP_BASE_DIR", raising=False)

    def resolve_root():
        configured = os.getenv("APP_BASE_DIR")
        return Path(configured).resolve() if configured else project_root

    monkeypatch.setattr(app_config, "get_project_root", resolve_root)

    config = app_config.load_config()

    assert config.model_registry_path == (configured_root / "models" / "registry.json").resolve()
    assert config.data_raw_dir == (configured_root / "data" / "raw").resolve()
