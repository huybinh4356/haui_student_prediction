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

def load_config() -> AppConfig:
    load_dotenv()
    
    base_dir = Path.cwd()
    
    return AppConfig(
        env=os.getenv("APP_ENV", "development"),
        data_raw_dir=base_dir / os.getenv("DATA_RAW_DIR", "data/raw"),
        data_processed_dir=base_dir / os.getenv("DATA_PROCESSED_DIR", "data/processed"),
        model_registry_path=base_dir / os.getenv("MODEL_REGISTRY_PATH", "models/registry.json"),
        model_artifacts_dir=base_dir / os.getenv("MODEL_ARTIFACTS_DIR", "models/artifacts"),
        report_output_dir=base_dir / "outputs" / "reports"
    )