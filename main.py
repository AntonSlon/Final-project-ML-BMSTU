from src.data import load_datasets, make_train_test_split, prepare_trans_log, prepare_p2p_log
from src.config import RAW_DATA_DIR
from src.features import build_features
from src.train import train_model
from src.utils import save_model, load_model, save_json
from sklearn.metrics import classification_report, confusion_matrix
from src.evaluate import evaluate_model
from src.threshold import find_best_threshold


def main() -> None:
    p2p_log, trans_log = load_datasets(
        RAW_DATA_DIR / "final_p2p_log.csv",
        RAW_DATA_DIR / "final_trans_log.csv"
    )

    p2p_log = prepare_p2p_log(p2p_log)
    trans_log = prepare_trans_log(trans_log)

    p2p_train, p2p_val, p2p_test, trans_train, trans_val, trans_test = make_train_test_split(p2p_log, trans_log, 0.8)
    X_train, y_train, cat_features = build_features(p2p_train, trans_train)

    """ раскоментить при первом запуске"""

    # model = train_model(X_train, y_train, cat_features)
    # save_model(model, "fraud_model_final.pkl")

    """ загрузка модели """
    model = load_model("fraud_model_final.pkl")

    X_val, y_val, cat_features = build_features(p2p_val, trans_val)
    y_val_score = model.predict_proba(X_val)[:, 1]
    precision, recall, threshold = find_best_threshold(y_true=y_val, y_score=y_val_score)

    X_test, y_test, cat_features = build_features(p2p_test, trans_test)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    report = evaluate_model(y_true=y_test, y_score=y_score, threshold=threshold, y_pred=y_pred)
    save_json(report, "metrics")

    print(classification_report(y_pred=y_pred, y_true=y_test))
    print(confusion_matrix(y_pred=y_pred, y_true=y_test))


if __name__ == "__main__":
    main()
