"""Unit tests for the ML pipeline & feature contract (Giai đoạn 3).

Covers: feature leakage, missing columns, pipeline fit/train-test split,
normal train/predict, and the no-model scenario. Uses small deterministic
fixtures -- no real student data.
"""
import pytest
import pandas as pd
import numpy as np

from haui_student_prediction.features.feature_contract import (
    FeatureContract,
    IDENTIFIER_COLUMNS,
)
from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline
from haui_student_prediction.ml.train_regressor import train_and_evaluate
from haui_student_prediction.ml.model_registry import ModelRegistry
from haui_student_prediction.scoring.final_course_score import (
    CourseProfile,
    ScoringService,
)
from haui_student_prediction.services.prediction_service import PredictionService


# ---- shared fixtures --------------------------------------------------------

NUM_COLS = ["absence_percentage", "tx1_score", "midterm_score", "lms_learning_percentage"]
CAT_COLS = ["works_part_time"]
TARGET = "actual_final_exam_score"


@pytest.fixture
def contract():
    return FeatureContract(
        numerical_features=NUM_COLS,
        categorical_features=CAT_COLS,
        target_column=TARGET,
    )


@pytest.fixture
def builder(contract):
    return FeatureBuilder(contract)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(
        registry_path=str(tmp_path / "registry.json"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )


@pytest.fixture
def scoring_service():
    profiles = {
        "no_tx2": CourseProfile(0.20, 0.0, 0.20, 0.60, False),
        "with_tx2": CourseProfile(0.15, 0.15, 0.20, 0.50, True),
    }
    return ScoringService(profiles)


@pytest.fixture
def prediction_service(registry, scoring_service):
    return PredictionService(registry, scoring_service)


def _make_train_df(n=20, seed=42):
    """Small synthetic training dataframe with a linear target."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "student_id": [f"SV{i:03d}" for i in range(n)],
        "full_name": [f"Name {i}" for i in range(n)],
        "absence_percentage": rng.uniform(0, 30, n),
        "lms_learning_percentage": rng.uniform(0, 100, n),
        "tx1_score": rng.uniform(5, 9, n),
        "midterm_score": rng.uniform(4, 9, n),
        "works_part_time": rng.choice([True, False], n),
        "actual_final_exam_score": rng.uniform(4, 9, n),
    })
    return df


def _make_test_df(n=5, seed=7):
    return _make_train_df(n=n, seed=seed)


# ---- 1. feature contract: leakage prevention -------------------------------

def test_contract_rejects_target_as_feature():
    with pytest.raises(ValueError, match="leakage"):
        FeatureContract(
            numerical_features=["tx1_score", TARGET],
            categorical_features=[],
            target_column=TARGET,
        )


def test_contract_rejects_identifier_as_feature():
    for ident in ["student_id", "full_name"]:
        with pytest.raises(ValueError, match="Identifier"):
            FeatureContract(
                numerical_features=[ident],
                categorical_features=[],
                target_column=TARGET,
            )


def test_contract_rejects_overlap_num_cat():
    with pytest.raises(ValueError, match="both numerical and categorical"):
        FeatureContract(
            numerical_features=["tx1_score"],
            categorical_features=["tx1_score"],
            target_column=TARGET,
        )


def test_contract_get_all_features_excludes_target(contract):
    feats = contract.get_all_features()
    assert TARGET not in feats
    assert set(feats) == set(NUM_COLS) | set(CAT_COLS)


# ---- 2. feature builder: leakage guards ------------------------------------

def test_builder_extracts_features_only(builder, contract):
    df = _make_train_df()
    X = builder.extract_features(df, is_training=True)
    assert TARGET not in X.columns
    for ident in IDENTIFIER_COLUMNS:
        assert ident not in X.columns
    assert set(X.columns) == set(contract.get_all_features())


def test_builder_extract_target_separate(builder):
    df = _make_train_df()
    y = builder.extract_target(df)
    assert y.name == TARGET
    assert len(y) == len(df)


def test_builder_missing_feature_raises(builder):
    df = _make_train_df().drop(columns=["tx1_score"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        builder.extract_features(df, is_training=True)


def test_builder_missing_target_raises(builder):
    df = _make_train_df().drop(columns=[TARGET])
    with pytest.raises(ValueError, match="Target column"):
        builder.extract_features(df, is_training=True)


def test_builder_blocks_target_even_if_in_df(builder):
    """Even if df contains the target, features must exclude it."""
    df = _make_train_df()
    X = builder.extract_features(df, is_training=False)
    assert TARGET not in X.columns


# ---- 3. preprocessing pipeline ---------------------------------------------

def test_preprocessing_rejects_empty_columns():
    with pytest.raises(ValueError, match="no numeric or categorical"):
        build_preprocessing_pipeline([], [])


def test_preprocessing_handles_unseen_category():
    """OneHotEncoder with handle_unknown='ignore' must not crash on unseen
    categories at transform time."""
    pipe = build_preprocessing_pipeline(["a"], ["cat"])
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "cat": ["x", "y", "x", "y"]})
    pipe.fit(X_train)
    # 'z' is unseen -- must not raise
    X_test = pd.DataFrame({"a": [1.5], "cat": ["z"]})
    out = pipe.transform(X_test)
    assert out.shape[0] == 1


# ---- 4. train_and_evaluate: normal train/predict ---------------------------

def test_train_evaluate_normal(contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    assert metrics.mae >= 0
    assert metrics.rmse >= 0
    # R² defined for n_test >= 2; may be any real number but must be finite.
    assert np.isfinite(metrics.r2)


def test_train_evaluate_uses_linear_regression(contract):
    """Requirement #3: model must be LinearRegression, not Ridge."""
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, _ = train_and_evaluate(df_train, df_test, contract)
    from sklearn.linear_model import LinearRegression
    assert isinstance(model.named_steps["regressor"], LinearRegression)


def test_train_evaluate_no_leakage_small_test(contract):
    """Pipeline must fit on train only; test set is transformed, not fit."""
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    # Predictions should be producible for test rows
    builder = FeatureBuilder(contract)
    X_test = builder.extract_features(df_test, is_training=False)
    preds = model.predict(X_test)
    assert len(preds) == 5


# ---- 5. train edge cases: too-small, empty target, empty test --------------

def test_train_too_few_rows_raises(contract):
    df = _make_train_df(n=2)
    with pytest.raises(ValueError, match="too small"):
        train_and_evaluate(df, df, contract)


def test_train_empty_train_raises(contract):
    with pytest.raises(ValueError, match="empty"):
        train_and_evaluate(pd.DataFrame(), _make_test_df(), contract)


def test_train_all_nan_target_raises(contract):
    df = _make_train_df(n=10)
    df[TARGET] = np.nan
    with pytest.raises(ValueError, match="entirely NaN"):
        train_and_evaluate(df, _make_test_df(), contract)


def test_train_empty_test_returns_nan_metrics(contract):
    df_train = _make_train_df(n=20)
    df_test = pd.DataFrame(columns=df_train.columns)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    assert np.isnan(metrics.mae)
    assert np.isnan(metrics.rmse)
    assert np.isnan(metrics.r2)


def test_train_single_test_point_r2_nan(contract):
    """R² undefined for n_test < 2 -- must be NaN, not fabricated."""
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=1)
    _, metrics = train_and_evaluate(df_train, df_test, contract)
    assert np.isnan(metrics.r2)
    assert np.isfinite(metrics.mae)   # MAE still valid for 1 point


# ---- 6. model registry: save/load roundtrip --------------------------------

def test_registry_save_load_roundtrip(registry, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)

    registry.save_model(model, "1.0.0", "hash123", metrics, contract)

    assert registry.has_model("1.0.0")
    loaded_model, meta = registry.load_model("1.0.0")
    assert "features" in meta
    assert meta["features"]["target_column"] == TARGET
    # Loaded model can predict
    builder = FeatureBuilder(contract)
    X = builder.extract_features(df_test, is_training=False)
    preds = loaded_model.predict(X)
    assert len(preds) == 5


def test_registry_has_model_false_when_unregistered(registry):
    assert not registry.has_model("1.0.0")


def test_registry_load_missing_version_raises(registry):
    with pytest.raises(ValueError, match="not registered"):
        registry.load_model("9.9.9")


def test_registry_get_contract_returns_contract(registry, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)
    c = registry.get_contract("1.0.0")
    assert c is not None
    assert c.target_column == TARGET


# ---- 7. prediction service: no-model scenario ------------------------------

def test_prediction_service_no_model_raises(prediction_service):
    """Requirement #7: when no model is registered, predict must raise a clear
    business error, not fabricate predictions."""
    df = _make_train_df(n=5)
    with pytest.raises(RuntimeError, match="not ready"):
        prediction_service.predict_student_results(df, "1.0.0", "no_tx2")


def test_prediction_service_is_model_ready_false(prediction_service):
    assert prediction_service.is_model_ready("1.0.0") is False


def test_prediction_service_is_model_ready_true_after_train(registry, scoring_service, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    svc = PredictionService(registry, scoring_service)
    assert svc.is_model_ready("1.0.0") is True


# ---- 8. prediction service: normal predict + schema check ------------------

def test_prediction_service_predict_normal(registry, scoring_service, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    svc = PredictionService(registry, scoring_service)
    result = svc.predict_student_results(df_test, "1.0.0", "no_tx2")
    assert "predicted_final_exam_score" in result.columns
    assert "predicted_course_score" in result.columns
    assert "target_final_exam_score" not in result.columns
    assert len(result) == 5
    # Predicted scores must be finite numbers
    assert result["predicted_final_exam_score"].notna().all()


def test_prediction_service_predict_missing_feature_raises(registry, scoring_service, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    svc = PredictionService(registry, scoring_service)
    # Remove a required feature column
    bad_df = df_test.drop(columns=["tx1_score"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        svc.predict_student_results(bad_df, "1.0.0", "no_tx2")


def test_prediction_service_empty_input_raises(registry, scoring_service, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    svc = PredictionService(registry, scoring_service)
    with pytest.raises(ValueError, match="empty"):
        svc.predict_student_results(pd.DataFrame(), "1.0.0", "no_tx2")


# ---- 9. feature leakage end-to-end: target must not leak into features -----

def test_no_feature_leakage_end_to_end(contract):
    """Requirement #1 & #6: the target column must never appear in the feature
    matrix used for training or prediction."""
    df = _make_train_df(n=20)
    builder = FeatureBuilder(contract)
    X = builder.extract_features(df, is_training=True)
    y = builder.extract_target(df)
    assert TARGET not in X.columns
    # X and y must be separate objects
    assert X is not y
    # No identifier columns in X
    for ident in IDENTIFIER_COLUMNS:
        assert ident not in X.columns


# ---- 10. registry corruption & missing artifact ----------------------------

def test_registry_corrupt_metadata_returns_false(registry, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    # Corrupt the registry entry by replacing with a string
    data = registry._load_registry()
    data["1.0.0"] = "corrupt"
    registry._save_registry(data)

    assert not registry.has_model("1.0.0")


def test_registry_missing_artifact_file_returns_false(registry, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    # Delete artifact file but keep registry entry
    artifact_path = registry.artifacts_dir / "model_v1.0.0.joblib"
    artifact_path.unlink()

    assert not registry.has_model("1.0.0")


def test_registry_load_missing_artifact_raises(registry, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    artifact_path = registry.artifacts_dir / "model_v1.0.0.joblib"
    artifact_path.unlink()

    with pytest.raises(FileNotFoundError, match="missing on disk"):
        registry.load_model("1.0.0")


def test_registry_missing_features_returns_false(registry, contract):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    data = registry._load_registry()
    del data["1.0.0"]["features"]
    registry._save_registry(data)

    assert not registry.has_model("1.0.0")


# ---- 11. training with missing target labels --------------------------------

def test_train_with_some_nan_target_ok(contract):
    """Rows with NaN target should be dropped from training, not crash."""
    df_train = _make_train_df(n=20)
    df_train.loc[0:2, TARGET] = np.nan
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    assert metrics.mae >= 0
    assert metrics.rmse >= 0


# ---- 12. prediction service unavailable model -------------------------------

def test_prediction_service_unavailable_after_artifact_deleted(
    registry, scoring_service, contract,
):
    df_train = _make_train_df(n=20)
    df_test = _make_test_df(n=5)
    model, metrics = train_and_evaluate(df_train, df_test, contract)
    registry.save_model(model, "1.0.0", "h", metrics, contract)

    svc = PredictionService(registry, scoring_service)
    assert svc.is_model_ready("1.0.0")

    artifact_path = registry.artifacts_dir / "model_v1.0.0.joblib"
    artifact_path.unlink()

    assert not svc.is_model_ready("1.0.0")
    with pytest.raises(RuntimeError, match="not ready"):
        svc.predict_student_results(df_test, "1.0.0", "no_tx2")
