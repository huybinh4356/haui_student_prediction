from typing import List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessing_pipeline(
    num_cols: List[str], cat_cols: List[str]
) -> ColumnTransformer:
    """Build a leakage-safe preprocessing ColumnTransformer.

    The transformer is designed to live INSIDE an sklearn Pipeline so that
    fit/transform only happens on the train split (requirement #2). It must be
    fit on train data and then transformed on test data -- never fit on the
    full dataset before splitting.

    Handles (requirement #4):
      * numeric columns: median imputation (handles missing values) + scaling.
      * categorical columns: most-frequent imputation + OneHotEncoder with
        handle_unknown='ignore' (handles unseen categories at predict time).
    """
    transformers = []

    if num_cols:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_transformer, num_cols))

    if cat_cols:
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_transformer, cat_cols))

    if not transformers:
        raise ValueError(
            "Cannot build preprocessing pipeline: no numeric or categorical columns provided"
        )

    return ColumnTransformer(transformers=transformers)
