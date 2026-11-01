import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from dataclasses import asdict
from haui_student_prediction.ml.train_regressor import ModelMetrics
from haui_student_prediction.features.feature_contract import FeatureContract


ACTIVE_MODEL_VERSION = "xgboost-2.2.0"


class ModelRegistry:
    """Single source of truth for trained model artifacts + metadata."""

    def __init__(self, registry_path: str, artifacts_dir: str):
        self.registry_path = Path(registry_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._save_registry({})

    def _load_registry(self) -> Dict[str, Any]:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, data: Dict[str, Any]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def save_model(
        self,
        model: Pipeline,
        version: str,
        dataset_hash: str,
        metrics: ModelMetrics,
        contract: FeatureContract,
    ) -> None:
        contract.validate()
        artifact_filename = f"model_v{version}.joblib"
        artifact_path = self.artifacts_dir / artifact_filename
        joblib.dump(model, artifact_path)

        registry = self._load_registry()
        registry[version] = {
            "version": version,
            "dataset_hash": dataset_hash,
            "metrics": asdict(metrics),
            "features": asdict(contract),
            "artifact_filename": artifact_filename,
        }
        self._save_registry(registry)

    def has_model(self, version: str) -> bool:
        try:
            registry = self._load_registry()
            if version not in registry:
                return False
            metadata = registry[version]
            if not isinstance(metadata, dict):
                return False
            artifact_name = metadata.get("artifact_filename")
            if not artifact_name:
                return False
            if not metadata.get("features"):
                return False
            artifact_path = self.artifacts_dir / artifact_name
            return artifact_path.exists()
        except Exception:
            return False

    def load_model(self, version: str) -> tuple[Pipeline, Dict[str, Any]]:
        registry = self._load_registry()
        if version not in registry:
            raise ValueError(
                f"Model version '{version}' is not registered. "
                "Train and save a model before requesting predictions."
            )

        metadata = registry[version]
        if not isinstance(metadata, dict):
            raise ValueError(f"Registry entry for '{version}' is corrupt")

        features = metadata.get("features")
        if not features or not isinstance(features, dict):
            raise ValueError(
                f"Registry entry for '{version}' is missing valid feature metadata"
            )

        artifact_name = metadata.get("artifact_filename")
        if not artifact_name:
            raise ValueError(
                f"Registry entry for '{version}' has no artifact_filename"
            )

        artifact_path = self.artifacts_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact file '{artifact_name}' is missing on disk"
            )

        model = joblib.load(artifact_path)
        return model, metadata

    def get_contract(self, version: str) -> Optional[FeatureContract]:
        try:
            _, metadata = self.load_model(version)
            return FeatureContract(**metadata["features"])
        except (ValueError, FileNotFoundError, TypeError, KeyError):
            return None
