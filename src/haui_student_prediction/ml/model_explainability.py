import numpy as np
import pandas as pd
from xgboost import DMatrix


def tree_feature_contributions(
    model,
    feature_dataframe: pd.DataFrame,
    source_features: list[str],
) -> pd.DataFrame:
    """Return TreeSHAP contributions grouped by original input feature.

    Each feature contribution is measured in predicted-score points. Positive
    values push a prediction above the model baseline; negative values push it
    below. One-hot encoded category contributions are grouped back to the
    original source column.
    """
    if feature_dataframe is None or feature_dataframe.empty:
        return pd.DataFrame(index=feature_dataframe.index if feature_dataframe is not None else None)

    if not hasattr(model, "named_steps"):
        raise TypeError("The trained model must be an sklearn Pipeline.")

    preprocessor = model.named_steps.get("preprocessor")
    regressor = model.named_steps.get("regressor")
    if preprocessor is None or regressor is None:
        raise ValueError("The trained pipeline is missing preprocessing or regressor steps.")

    transformed = preprocessor.transform(feature_dataframe)
    transformed_names = list(preprocessor.get_feature_names_out())
    booster = regressor.get_booster()
    raw_contributions = np.asarray(
        booster.predict(DMatrix(transformed), pred_contribs=True),
        dtype=float,
    )

    if raw_contributions.ndim != 2 or raw_contributions.shape[1] != len(transformed_names) + 1:
        raise RuntimeError("XGBoost returned an unexpected TreeSHAP contribution shape.")

    grouped = pd.DataFrame(
        0.0,
        index=feature_dataframe.index,
        columns=source_features,
    )
    ordered_sources = sorted(source_features, key=len, reverse=True)

    for position, transformed_name in enumerate(transformed_names):
        source_name = transformed_name.split("__", 1)[-1]
        matched_source = next(
            (
                source
                for source in ordered_sources
                if source_name == source or source_name.startswith(source + "_")
            ),
            None,
        )
        if matched_source is None:
            raise RuntimeError(
                f"Cannot map transformed feature '{transformed_name}' "
                "back to its source column."
            )
        grouped[matched_source] += raw_contributions[:, position]

    grouped["__baseline__"] = raw_contributions[:, -1]
    expected = np.asarray(model.predict(feature_dataframe), dtype=float).reshape(-1)
    reconstructed = (
        grouped[source_features].sum(axis=1).to_numpy(dtype=float)
        + grouped["__baseline__"].to_numpy(dtype=float)
    )
    if not np.allclose(reconstructed, expected, rtol=1e-4, atol=1e-4):
        raise RuntimeError(
            "TreeSHAP contributions do not reconstruct the model's predicted score."
        )

    return grouped
