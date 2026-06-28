import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task

from src.pipeline import (
    evaluate_model_artifact,
    prepare_training_features,
    select_model_threshold,
    train_model_artifact,
    validate_training_inputs,
)


DEFAULT_ARGS = {
    "owner": "fraud-ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="fraud_model_training",
    description="Train, validate and evaluate the P2P fraud model",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["fraud", "training", "catboost"],
)
def fraud_model_training_dag():
    @task
    def validate_data():
        return validate_training_inputs(
            p2p_path=os.getenv(
                "P2P_DATA_PATH", "/opt/airflow/data/raw/final_p2p_log.csv"
            ),
            trans_path=os.getenv(
                "TRANS_DATA_PATH", "/opt/airflow/data/raw/final_trans_log.csv"
            ),
        )

    @task
    def generate_features(input_paths):
        return prepare_training_features(input_paths)

    @task
    def train_model(feature_artifacts):
        return train_model_artifact(feature_artifacts)

    @task
    def select_threshold(model_artifacts):
        return select_model_threshold(model_artifacts)

    @task
    def evaluate_model(threshold_artifacts):
        return evaluate_model_artifact(threshold_artifacts)

    inputs = validate_data()
    features = generate_features(inputs)
    model = train_model(features)
    threshold = select_threshold(model)
    evaluate_model(threshold)


fraud_model_training_dag()
