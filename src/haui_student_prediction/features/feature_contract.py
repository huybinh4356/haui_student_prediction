from dataclasses import dataclass
from typing import List

@dataclass
class FeatureContract:
    numerical_features: List[str]
    categorical_features: List[str]
    target_column: str

    def get_all_features(self) -> List[str]:
        return self.numerical_features + self.categorical_features