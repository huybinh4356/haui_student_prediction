import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class AppConfig:
    env: str
    data_raw_dir: Path
    data_processed_dir: Path
    model_registry_path: Path
    model_artifacts_dir: Path
    report_output_dir: Path

def get_project_root() -> Path:
    env_root = os.getenv("APP_BASE_DIR")
    if env_root:
        return Path(env_root).resolve()
    # app_config.py is at <root>/src/haui_student_prediction/app_config.py
    return Path(__file__).resolve().parent.parent.parent

def _resolve_path(base_dir: Path, configured_path: str) -> Path:
    p = Path(configured_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()

def load_config() -> AppConfig:
    base_dir = get_project_root()
    env_file = base_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    return AppConfig(
        env=os.getenv("APP_ENV", "development"),
        data_raw_dir=_resolve_path(
            base_dir, os.getenv("DATA_RAW_DIR", "data/raw"),
        ),
        data_processed_dir=_resolve_path(
            base_dir, os.getenv("DATA_PROCESSED_DIR", "data/processed"),
        ),
        model_registry_path=_resolve_path(
            base_dir, os.getenv("MODEL_REGISTRY_PATH", "models/registry.json"),
        ),
        model_artifacts_dir=_resolve_path(
            base_dir, os.getenv("MODEL_ARTIFACTS_DIR", "models/artifacts"),
        ),
        report_output_dir=_resolve_path(
            base_dir, os.getenv("REPORT_OUTPUT_DIR", "outputs/reports"),
        ),
    )
