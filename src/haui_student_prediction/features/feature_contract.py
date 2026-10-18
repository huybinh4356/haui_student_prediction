from dataclasses import dataclass
from typing import List

# Columns that must NEVER appear in features:
#   - identifiers (leak unique info, no generalisation)
#   - target column (direct leakage)
IDENTIFIER_COLUMNS = ["student_id", "full_name", "snapshot_date", "date_of_birth"]


@dataclass
class FeatureContract:
    numerical_features: List[str]
    categorical_features: List[str]
    target_column: str

    def __post_init__(self) -> None:
        self.validate()

    def get_all_features(self) -> List[str]:
        """Return features only (numerical + categorical), never the target."""
        return self.numerical_features + self.categorical_features

    def validate(self) -> None:
        """Validate the contract itself: no identifier/target in features, no
        overlap between numerical and categorical lists, no empty target."""
        all_feats = self.get_all_features()
        if not all_feats:
            raise ValueError("FeatureContract has no features defined")
        if not self.target_column:
            raise ValueError("FeatureContract target_column is empty")

        overlap = set(self.numerical_features) & set(self.categorical_features)
        if overlap:
            raise ValueError(
                f"Columns appear in both numerical and categorical features: {sorted(overlap)}"
            )

        leaked_ids = set(all_feats) & set(IDENTIFIER_COLUMNS)
        if leaked_ids:
            raise ValueError(
                f"Identifier columns must not be features: {sorted(leaked_ids)}"
            )

        if self.target_column in all_feats:
            raise ValueError(
                f"Target column '{self.target_column}' must not be a feature (leakage)"
            )
