from src.config import METRICS_PATH, MODEL_PATH, REPORTS_DIR
from catboost import CatBoostClassifier
import json


def save_model(model, name):
    model.save_model(MODEL_PATH / name)


def load_model(name):
    loaded_model = CatBoostClassifier()
    return loaded_model.load_model(MODEL_PATH / name)


def save_json(data: dict, name) -> None:
    with open(REPORTS_DIR / f"{name}.json", "w", encoding="utf-8") as file:
        json.dump(data, file)

