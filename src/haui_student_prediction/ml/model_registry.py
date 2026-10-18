import json
import joblib
from pathlib import Path
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from dataclasses import asdict
from haui_student_prediction.ml.train_regressor import ModelMetrics
from haui_student_prediction.features.feature_contract import FeatureContract

class ModelRegistry:
    def __init__(self, registry_path: str, artifacts_dir: str):
        self.registry_path = Path(registry_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.registry_path.exists():
            self._save_registry({})

    def _load_registry(self) -> Dict[str, Any]:
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_registry(self, data: Dict[str, Any]) -> None:
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save_model(
        self, 
        model: Pipeline, 
        version: str, 
        dataset_hash: str, 
        metrics: ModelMetrics, 
        contract: FeatureContract
    ) -> None:
        artifact_filename = f"model_v{version}.joblib"
        artifact_path = self.artifacts_dir / artifact_filename
        
        joblib.dump(model, artifact_path)

        registry = self._load_registry()
        registry[version] = {
            "version": version,
            "dataset_hash": dataset_hash,
            "metrics": asdict(metrics),
            "features": asdict(contract),
            "artifact_filename": artifact_filename
        }
        self._save_registry(registry)

    def load_model(self, version: str) -> tuple[Pipeline, Dict[str, Any]]:
        registry = self._load_registry()
        if version not in registry:
            raise ValueError("Model version not found")

        metadata = registry[version]
        artifact_path = self.artifacts_dir / metadata["artifact_filename"]
        
        if not artifact_path.exists():
            raise FileNotFoundError("Model artifact file is missing")

        model = joblib.load(artifact_path)
        return model, metadata