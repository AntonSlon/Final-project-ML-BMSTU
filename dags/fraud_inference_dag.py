import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task

from src.pipeline import (
    predict_inference_batch,
    prepare_inference_features,
    save_inference_predictions,
    validate_inference_inputs,
)


DEFAULT_ARGS = {
    "owner": "fraud-ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="fraud_batch_inference",
    description="Score a batch of new P2P transfers",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["fraud", "inference", "batch"],
)
def fraud_batch_inference_dag():
    @task
    def validate_inputs():
        return validate_inference_inputs(
            new_p2p_path=os.getenv(
                "INFERENCE_P2P_PATH",
                "/opt/airflow/data/inference/new_p2p.csv",
            ),
            history_p2p_path=os.getenv(
                "P2P_DATA_PATH", "/opt/airflow/data/raw/final_p2p_log.csv"
            ),
            history_trans_path=os.getenv(
                "TRANS_DATA_PATH", "/opt/airflow/data/raw/final_trans_log.csv"
            ),
        )

    @task
    def generate_features(input_paths):
        return prepare_inference_features(input_paths)

    @task
    def predict(feature_artifacts):
        return predict_inference_batch(feature_artifacts)

    @task
    def save_predictions(scored_artifacts):
        return save_inference_predictions(
            scored_artifacts,
            output_path=os.getenv(
                "PREDICTIONS_PATH",
                "/opt/airflow/data/predictions/fraud_predictions.csv",
            ),
        )

    inputs = validate_inputs()
    features = generate_features(inputs)
    scored = predict(features)
    save_predictions(scored)


fraud_batch_inference_dag()
