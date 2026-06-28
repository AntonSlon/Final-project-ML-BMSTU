import json
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from src.data import (
    load_datasets,
    make_train_test_split,
    prepare_p2p_log,
    prepare_trans_log,
)
from src.evaluate import evaluate_model
from src.features import build_features
from src.threshold import find_best_threshold
from src.train import train_model
from src.utils import load_model, save_json, save_model


LOGGER = logging.getLogger(__name__)

MODEL_NAME = os.getenv("FRAUD_MODEL_NAME", "fraud_model_no_leakage.cbm")
THRESHOLD_PATH = MODELS_DIR / "fraud_threshold.json"
TARGET_COLUMN = "__target__"

TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / "train_features.pkl"
VALIDATION_FEATURES_PATH = PROCESSED_DATA_DIR / "validation_features.pkl"
TEST_FEATURES_PATH = PROCESSED_DATA_DIR / "test_features.pkl"

INFERENCE_FEATURES_PATH = PROCESSED_DATA_DIR / "inference_features.pkl"
INFERENCE_ROWS_PATH = PROCESSED_DATA_DIR / "inference_rows.pkl"
SCORED_ROWS_PATH = PROCESSED_DATA_DIR / "scored_rows.pkl"


def _require_file(path, description):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")
    return path


def _read_p2p_file(path, nrows=None):
    dataframe = pd.read_csv(path, nrows=nrows)
    unnamed_columns = [
        column for column in dataframe.columns if column.startswith("Unnamed:")
    ]
    return dataframe.drop(columns=unnamed_columns, errors="ignore")


def _save_feature_dataset(features, target, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = features.copy()
    dataset[TARGET_COLUMN] = target.to_numpy()
    dataset.to_pickle(path)
    LOGGER.info("Saved %s rows to %s", len(dataset), path)
    return str(path)


def _load_feature_dataset(path):
    dataset = pd.read_pickle(path)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Target column is missing in feature artifact: {path}")
    target = dataset.pop(TARGET_COLUMN)
    return dataset, target


def _save_threshold(threshold):
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_PATH.open("w", encoding="utf-8") as file:
        json.dump({"threshold": float(threshold)}, file, indent=2)
    return str(THRESHOLD_PATH)


def _load_threshold():
    _require_file(THRESHOLD_PATH, "Threshold artifact")
    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        return float(json.load(file)["threshold"])


def validate_training_inputs(
    p2p_path=RAW_DATA_DIR / "final_p2p_log.csv",
    trans_path=RAW_DATA_DIR / "final_trans_log.csv",
):
    p2p_path = _require_file(p2p_path, "P2P training dataset")
    trans_path = _require_file(trans_path, "Transaction training dataset")

    p2p_sample = prepare_p2p_log(_read_p2p_file(p2p_path, nrows=1000))
    trans_sample = prepare_trans_log(pd.read_csv(trans_path, nrows=1000))
    if "IsFraud" not in p2p_sample.columns:
        raise ValueError("Training P2P dataset must contain IsFraud")

    LOGGER.info(
        "Training inputs validated: p2p_columns=%s, trans_columns=%s",
        list(p2p_sample.columns),
        list(trans_sample.columns),
    )
    return {"p2p_path": str(p2p_path), "trans_path": str(trans_path)}


def prepare_training_features(input_paths):
    p2p_log, trans_log = load_datasets(
        input_paths["p2p_path"],
        input_paths["trans_path"],
    )
    p2p_log = prepare_p2p_log(p2p_log)
    trans_log = prepare_trans_log(trans_log)

    (
        p2p_train,
        p2p_val,
        p2p_test,
        trans_train,
        trans_val,
        _,
    ) = make_train_test_split(p2p_log, trans_log, 0.8)

    history_end = p2p_train.iloc[int(len(p2p_train) * 0.2)]["EventTime"]
    p2p_history = p2p_train[p2p_train["EventTime"] < history_end].copy()
    p2p_model_train = p2p_train[
        p2p_train["EventTime"] >= history_end
    ].copy()
    trans_history = trans_train[
        trans_train["EventTime"] < history_end
    ].copy()

    X_train, y_train, cat_features = build_features(
        p2p_model_train,
        p2p_history,
        trans_history,
    )
    X_val, y_val, _ = build_features(
        p2p_val,
        p2p_train,
        trans_train,
    )

    p2p_test_history = pd.concat([p2p_train, p2p_val])
    trans_test_history = pd.concat([trans_train, trans_val])
    X_test, y_test, _ = build_features(
        p2p_test,
        p2p_test_history,
        trans_test_history,
    )

    artifacts = {
        "train_features": _save_feature_dataset(
            X_train, y_train, TRAIN_FEATURES_PATH
        ),
        "validation_features": _save_feature_dataset(
            X_val, y_val, VALIDATION_FEATURES_PATH
        ),
        "test_features": _save_feature_dataset(
            X_test, y_test, TEST_FEATURES_PATH
        ),
        "cat_features": cat_features,
    }
    LOGGER.info("Training feature artifacts prepared: %s", artifacts)
    return artifacts


def train_model_artifact(artifacts):
    X_train, y_train = _load_feature_dataset(artifacts["train_features"])
    model = train_model(X_train, y_train, artifacts["cat_features"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, MODEL_NAME)
    model_path = MODELS_DIR / MODEL_NAME

    result = dict(artifacts)
    result["model_name"] = MODEL_NAME
    result["model_path"] = str(model_path)
    LOGGER.info("Model saved to %s", model_path)
    return result


def select_model_threshold(artifacts):
    X_val, y_val = _load_feature_dataset(artifacts["validation_features"])
    model = load_model(artifacts["model_name"])
    scores = model.predict_proba(X_val)[:, 1]
    precision, recall, threshold = find_best_threshold(y_val, scores)
    precision = float(precision)
    recall = float(recall)
    threshold = float(threshold)
    threshold_path = _save_threshold(threshold)

    result = dict(artifacts)
    result.update(
        {
            "threshold": threshold,
            "threshold_path": threshold_path,
            "validation_precision": precision,
            "validation_recall": recall,
        }
    )
    LOGGER.info(
        "Threshold selected: threshold=%.6f precision=%.4f recall=%.4f",
        threshold,
        precision,
        recall,
    )
    return result


def evaluate_model_artifact(artifacts):
    X_test, y_test = _load_feature_dataset(artifacts["test_features"])
    model = load_model(artifacts["model_name"])
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= artifacts["threshold"]).astype(int)
    report = evaluate_model(
        y_true=y_test,
        y_score=scores,
        y_pred=predictions,
        threshold=artifacts["threshold"],
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(report, "metrics")
    metrics_path = REPORTS_DIR / "metrics.json"
    LOGGER.info("Test report:\n%s", classification_report(y_test, predictions))
    LOGGER.info("Metrics saved to %s", metrics_path)
    return str(metrics_path)


def validate_inference_inputs(
    new_p2p_path=DATA_DIR / "inference" / "new_p2p.csv",
    history_p2p_path=RAW_DATA_DIR / "final_p2p_log.csv",
    history_trans_path=RAW_DATA_DIR / "final_trans_log.csv",
):
    new_p2p_path = _require_file(new_p2p_path, "Inference P2P batch")
    history_p2p_path = _require_file(history_p2p_path, "P2P history")
    history_trans_path = _require_file(history_trans_path, "Transaction history")
    _require_file(MODELS_DIR / MODEL_NAME, "Fraud model")
    _require_file(THRESHOLD_PATH, "Threshold artifact")

    new_sample = prepare_p2p_log(_read_p2p_file(new_p2p_path, nrows=1000))
    if new_sample.empty:
        raise ValueError("Inference batch is empty")

    LOGGER.info("Inference inputs validated: %s rows sampled", len(new_sample))
    return {
        "new_p2p_path": str(new_p2p_path),
        "history_p2p_path": str(history_p2p_path),
        "history_trans_path": str(history_trans_path),
    }


def prepare_inference_features(input_paths):
    p2p_history, trans_history = load_datasets(
        input_paths["history_p2p_path"],
        input_paths["history_trans_path"],
    )
    p2p_history = prepare_p2p_log(p2p_history)
    trans_history = prepare_trans_log(trans_history)
    current_p2p = prepare_p2p_log(
        _read_p2p_file(input_paths["new_p2p_path"])
    )

    current_start = current_p2p["EventTime"].min()
    p2p_history = p2p_history[
        p2p_history["EventTime"] < current_start
    ].copy()
    trans_history = trans_history[
        trans_history["EventTime"] < current_start
    ].copy()
    if p2p_history.empty or trans_history.empty:
        raise ValueError("No historical data is available before inference batch")

    features, _, _ = build_features(
        current_p2p,
        p2p_history,
        trans_history,
    )
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_pickle(INFERENCE_FEATURES_PATH)
    current_p2p.to_pickle(INFERENCE_ROWS_PATH)

    LOGGER.info("Prepared %s inference rows", len(features))
    return {
        "features_path": str(INFERENCE_FEATURES_PATH),
        "rows_path": str(INFERENCE_ROWS_PATH),
    }


def predict_inference_batch(artifacts):
    features = pd.read_pickle(artifacts["features_path"])
    rows = pd.read_pickle(artifacts["rows_path"])
    model = load_model(MODEL_NAME)
    threshold = _load_threshold()

    if model.feature_names_:
        features = features[model.feature_names_]
    scores = model.predict_proba(features)[:, 1]
    rows["fraud_score"] = scores
    rows["is_fraud_pred"] = (scores >= threshold).astype(int)
    rows.to_pickle(SCORED_ROWS_PATH)

    LOGGER.info(
        "Scored %s rows; fraud alerts=%s",
        len(rows),
        int(rows["is_fraud_pred"].sum()),
    )
    return {"scored_rows_path": str(SCORED_ROWS_PATH)}


def save_inference_predictions(
    artifacts,
    output_path=DATA_DIR / "predictions" / "fraud_predictions.csv",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_rows = pd.read_pickle(artifacts["scored_rows_path"])
    scored_rows.to_csv(output_path, index=False)
    LOGGER.info("Predictions saved to %s", output_path)
    return str(output_path)
