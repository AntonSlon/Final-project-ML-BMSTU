from catboost import CatBoostClassifier, CatBoostRegressor
from src.config import RANDOM_STATE


def train_model(X_train, y_train, cat_features):
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=True,
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    return model
