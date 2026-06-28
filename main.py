import argparse
import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODELS_DIR, RAW_DATA_DIR
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


MODEL_NAME = "fraud_model_no_leakage.cbm"
THRESHOLD_PATH = MODELS_DIR / "fraud_threshold.json"


def save_threshold(threshold: float) -> None:
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as file:
        json.dump({"threshold": float(threshold)}, file, indent=2)


def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            "Threshold is not saved. Run `python main.py train` first."
        )

    with open(THRESHOLD_PATH, encoding="utf-8") as file:
        return float(json.load(file)["threshold"])


def main(mode: str = "evaluate") -> None:
    p2p_log, trans_log = load_datasets(
        RAW_DATA_DIR / "final_p2p_log.csv",
        RAW_DATA_DIR / "final_trans_log.csv",
    )

    p2p_log = prepare_p2p_log(p2p_log)
    trans_log = prepare_trans_log(trans_log)

    (
        p2p_train,
        p2p_val,
        p2p_test,
        trans_train,
        trans_val,
        trans_test,
    ) = make_train_test_split(p2p_log, trans_log, 0.8)

    if mode == "train":
        history_end = p2p_train.iloc[int(len(p2p_train) * 0.2)]["EventTime"]
        p2p_history = p2p_train[
            p2p_train["EventTime"] < history_end
        ].copy()
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
        model = train_model(X_train, y_train, cat_features)

        X_val, y_val, _ = build_features(
            p2p_val,
            p2p_train,
            trans_train,
        )
        y_val_score = model.predict_proba(X_val)[:, 1]
        precision, recall, threshold = find_best_threshold(
            y_true=y_val,
            y_score=y_val_score,
        )

        save_model(model, MODEL_NAME)
        save_threshold(threshold)
        print(
            f"Validation precision={precision:.4f}, "
            f"recall={recall:.4f}, threshold={threshold:.4f}"
        )
    else:
        model = load_model(MODEL_NAME)
        threshold = load_threshold()

    p2p_test_history = pd.concat([p2p_train, p2p_val])
    trans_test_history = pd.concat([trans_train, trans_val])
    X_test, y_test, _ = build_features(
        p2p_test,
        p2p_test_history,
        trans_test_history,
    )
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    report = evaluate_model(
        y_true=y_test,
        y_score=y_score,
        threshold=threshold,
        y_pred=y_pred,
    )
    save_json(report, "metrics")

    print(classification_report(y_true=y_test, y_pred=y_pred))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the fraud detection model"
    )
    parser.add_argument(
        "mode",
        choices=["train", "evaluate"],
        default="evaluate",
        nargs="?",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.mode)
