import os
from pathlib import Path


PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
REPOSITORY = os.environ.get("BUCKET_NAME")  # Match the Bucket name on Artifact Registry
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# Define the pipeline name
PIPELINE_NAME = Path(__file__).resolve().parents[1].name


# Defautl arguments for the pipeline
default_params = {
    "project_id": os.environ.get("PROJECT_ID"),
    "region": os.environ.get("REGION"),
    "bucket_name": os.environ.get("BUCKET_NAME"),
    "pipeline_name": PIPELINE_NAME,
}

# ------------- Define the details component -------------
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{PIPELINE_NAME}:latest"
TRANSACTION_USAGE_FLAG = 50  # Number of transactions
TARGET_BALANCE = 20000  # CZK
SENIORITY_ACCOUNT_FLAG = 30
INCIDENT_DURATION_DAYS = 20
OFF_SET_PERIOD_DAYS = 365  # Period to collect for training
COMPUTE_TARGET = True
MODEL_NAME = "berkamodel"
SERVING_CONTAINER_PORTS = 8080
params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
}
SCORING = "roc_auc"
PROMOTION_PERFORMANCE = 0.7
MACHINE_TYPE = "n1-standard-2"
FILE_SCORING_NAME = "training_drivers"

eod_balance_component_args = {
    "column_mapping": {
        "transaction_date": "transaction_date",
        "account_id": "account_id",
        "transaction_amount": "transaction_amount",
        "transaction_type": "transaction_type",
        "daily_amount_flow": "daily_amount_flow",
        "end_of_day_balance": "end_of_day_balance",
        "account_creation_date": "account_creation_date",
        "balance_date": "balance_date",
        "outflow": "outflow",
    },
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/01_raw/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/02_elaboration/",
    "input_files": ["accounts", "transactions"],
    "output_files": ["eod_balance"],
}

preprocess_eod_balance_component_args = {
    "column_mapping": {
        "balance_date": "balance_date",
        "account_creation_date": "account_creation_date",
        "account_id": "account_id",
        "daily_amount_flow": "daily_amount_flow",
        "n_transactions": "n_transactions",
        "is_primary": "is_primary",
        "end_of_day_balance": "end_of_day_balance",
        "low_balance_flag": "low_balance_flag",
        "streak_id": "streak_id",
        "low_balance_streak": "low_balance_streak",
        "target": "target",
    },
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/02_elaboration/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/03_primary/",
    "input_files": ["eod_balance"],
    "output_files": ["eod_balance_preprocessed"],
    # Specific
    "transaction_usage_flag": TRANSACTION_USAGE_FLAG,
    "seniority_account_flag": SENIORITY_ACCOUNT_FLAG,
    "target_balance": TARGET_BALANCE,
    "incident_duration_days": INCIDENT_DURATION_DAYS,
    "compute_target": True,
}


aggregation_eod_balance_component_args = {
    "column_mapping": {
        "account_id": "account_id",
        "balance_date": "balance_date",
        "end_of_day_balance": "end_of_day_balance",
        "daily_amount_flow": "daily_amount_flow",
        "account_creation_date": "account_creation_date",
        "n_transactions": "n_transactions",
        "days_since_account_creation": "days_since_account_creation",
        "is_primary": "is_primary",
        "low_balance_flag": "low_balance_flag",
        "streak_id": "streak_id",
        "low_balance_streak": "low_balance_streak",
        "target": "target",
        "district_id": "district_id",
        "incident_date": "incident_date",
        "t0": "t0",
        "t1": "t1",
    },
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/03_primary/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/04_processing/",
    "input_files": ["eod_balance_preprocessed", "accounts"],
    "output_files": ["eod_balance_aggregation"],
    # Specific
    "incident_duration_days": INCIDENT_DURATION_DAYS,
    "off_set_period_days": OFF_SET_PERIOD_DAYS,
}

incidents_component_args = {
    "column_mapping": {
        "account_id": "account_id",
        "balance_date": "balance_date",
        "end_of_day_balance": "end_of_day_balance",
        "daily_amount_flow": "daily_amount_flow",
        "account_creation_date": "account_creation_date",
        "n_transactions": "n_transactions",
        "days_since_account_creation": "days_since_account_creation",
        "is_primary": "is_primary",
        "low_balance_flag": "low_balance_flag",
        "streak_id": "streak_id",
        "low_balance_streak": "low_balance_streak",
        "target": "target",
        "district_id": "district_id",
        "incident_date": "incident_date",
        "t0": "t0",
        "t1": "t1",
    },
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/03_primary/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/03_primary/",
    "input_files": ["eod_balance_preprocessed", "accounts"],
    "output_files": ["incidents"],
    # Specific
    "incident_duration_days": INCIDENT_DURATION_DAYS,
    "off_set_period_days": OFF_SET_PERIOD_DAYS,
}


eod_balance_training_args = {
    "column_mapping": {
        "account_id": "account_id",
        "balance_date": "balance_date",
        "end_of_day_balance": "end_of_day_balance",
        "daily_amount_flow": "daily_amount_flow",
        "account_creation_date": "account_creation_date",
        "n_transactions": "n_transactions",
        "days_since_account_creation": "days_since_account_creation",
        "is_primary": "is_primary",
        "low_balance_flag": "low_balance_flag",
        "streak_id": "streak_id",
        "low_balance_streak": "low_balance_streak",
        "target": "target",
        "district_id": "district_id",
        "incident_date": "incident_date",
        "t0": "t0",
        "t1": "t1",
    },
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/04_processing/",
    "input_files": {
        "eod_balance_aggregation": "eod_balance_aggregation",
        "accounts": "accounts",
        "incidents": "incidents",
    },
    "output_files": {
        "eod_balance_training": "eod_balance_training",
        "core_training": "core_training",
    },
}

features_enigneering_component_args = {
    "column_mapping": {
        "incidents": {
            "account_id": "account_id",
            "incident_date": "incident_date",
            "district_id": "district_id",
            "t0": "t0",
            "t1": "t1",
        },
        "eod_balance_training": {
            "account_id": "account_id",
            "balance_date": "balance_date",
            "end_of_day_balance": "end_of_day_balance",
            "daily_amount_flow": "daily_amount_flow",
            "n_transactions": "n_transactions",
            "days_since_account_creation": "days_since_account_creation",
            "low_balance_streak": "low_balance_streak",
            "district_id": "district_id",
        },
    },
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/05_features/",
    "input_files": {
        "eod_balance_training": "eod_balance_training",
        "accounts": "accounts",
        "incidents": "incidents",
    },
    "output_files": {"training_features": "training_features"},
    "aggregations": {
        "time_periods_days": [str(i) for i in range(1, 31)],
        "functions": ["mean", "std", "min", "max", "sum"],
        "columns": ["end_of_day_balance", "daily_amount_flow", "n_transactions"],
    },
}


feature_selection_component_args = {
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/05_features/",
    "input_files": {
        "training_features": "training_features",
        "core_training": "core_training",
    },
    "output_files": {"training_drivers": "training_drivers"},
    "step": 0.2,
    "cv": 10,
    "scoring": "roc_auc",
    "n_jobs": -1,
    "standard_error_threshold": 0.5,
    "return_type": "feature_names",
    "num_features": "best_coherent",
    "params": params,
}

hyperparameter_tuning_component_args = {
    "input_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/",
    "output_bucket_path": f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/data/05_features/",
    "input_files": {
        "training_drivers": "training_drivers",
        "core_training": "core_training",
    },
    "output_files": {},
    "params": params,
    "n_splits": 5,
    "n_iter": 10,
    "scoring": SCORING,
    "model_name": MODEL_NAME,
}

model_registry_component_args = {
    "model_name": MODEL_NAME,
    "serving_naive_runtime_container_image": BASE_IMAGE,
    "is_default_version": True,
    "version_aliases": [MODEL_NAME, "challanger", "random-forest"],
    "version_description": "Random Forest Model Classifier",
    "serving_container_ports": SERVING_CONTAINER_PORTS,
    "serving_container_health_route": "/health",
    "serving_container_predict_route": "/predict",
    "labels": {
        "createdby": "davide",
        "pipeline": PIPELINE_NAME,
    },
    "project_id": PROJECT_ID,
    "region": REGION,
}


batch_prediction_component_args = {
    "project_id": PROJECT_ID,
    "region": REGION,
    "model_resource_name": MODEL_NAME,
    "job_display_name": f"{MODEL_NAME}_batch_prediction_job",
    "gcs_source": f"gs://{BUCKET_NAME}/{PIPELINE_NAME}/data/05_features/{FILE_SCORING_NAME}.csv",
    "gcs_destination": f"gs://{BUCKET_NAME}/{PIPELINE_NAME}/data/07_output/",
    "instances_format": "csv",
    "machine_type": MACHINE_TYPE,
    "accelerator_count": 0,
    "starting_replica_count": 1,
    "max_replica_count": 1,
    "sync": False,
}


training_trigger_component_args = {
    "input_bucket_path": f"gs://{BUCKET_NAME}/{PIPELINE_NAME}/data/",
    "trigger_performance": PROMOTION_PERFORMANCE,
}
