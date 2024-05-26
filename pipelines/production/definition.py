import os
from pathlib import Path
from kfp import compiler, dsl
from utils.vertexai import vertex_authenticate


# Import components
from components.eod_balance.definition import create_eod_balance_component
from components.balance_processing.definition import process_eod_balance_component
from components.balance_aggregation.definition import eod_balance_aggregation_component
from components.incidents.definition import incidents_component
from components.core_training.definition import core_fair_eod_balance_training_component
from components.feature_engineering.definition import feature_engineering_component
from components.feature_selection.definition import feature_selection_component
from components.hyperparameter_tuning.definition import hyperparameter_tuning_component
from components.model_upload.definition import modelregistry_component

import datetime


# Authenticate with Google Cloud SDK for Vertex AI
aiplatform_client = vertex_authenticate()
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
REPOSITORY = os.environ.get("BUCKET_NAME")  # Match the Bucket name on Artifact Registry
# Define the pipeline name
PIPELINE_NAME = Path(__file__).resolve().parents[0].name

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
PROMOTION_PERFORMANCE = 0.3  # TODO Change to real value:0.8
MACHINE_TYPE = "n1-standard-2"

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
    "location": REGION,
    "model_resource_name": "model_resource_name_value",
    "job_display_name": "job_display_name_value",
    "gcs_source": ["gcs_source_value"],
    "gcs_destination": "gcs_destination_value",
    "instances_format": "instances_format_value",
    "machine_type": "machine_type_value",
    "accelerator_count": 0,
    "accelerator_type": "accelerator_type_value",
    "starting_replica_count": 1,
    "max_replica_count": 1,
    "sync": True,
}
# --------------------------------------------------------


# Define the pipeline
@dsl.pipeline(
    pipeline_root=f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/run/",
    name=PIPELINE_NAME,
    description=f"Pipeline on Vertex AI for {PIPELINE_NAME}",
)
def pipeline():

    # Create EOD Balance
    create_eod_balance_task = create_eod_balance_component(**eod_balance_component_args)

    # Process EOD Balance
    preprocess_eod_balance_task = process_eod_balance_component(
        **preprocess_eod_balance_component_args,
        eod_balance=create_eod_balance_task.outputs["eod_balance_output"],
    )

    # TODO: implement decision component to kick training if performance is under threshold
    # If EndPoint Model exists, check performance
    # If performance is under threshold, kick training
    # Else fetch model and score

    # Create incidents
    create_incidents_task = incidents_component(
        **incidents_component_args,
        eod_balance_preprocessed=preprocess_eod_balance_task.outputs[
            "eod_balance_preprocessed_output"
        ],
    )

    # Compute Aggregation for training - Fair EOD Balance Period
    aggregation_eod_balance_task = eod_balance_aggregation_component(
        **aggregation_eod_balance_component_args,
        eod_balance_preprocessed=preprocess_eod_balance_task.outputs[
            "eod_balance_preprocessed_output"
        ],
    )

    # Create EODB Training
    core_fair_eod_balance_training_task = core_fair_eod_balance_training_component(
        **eod_balance_training_args,
        eod_balance_aggregation=aggregation_eod_balance_task.outputs[
            "eod_balance_aggregation_output"
        ],
    )

    # Features Training
    features_engineering_task = feature_engineering_component(
        **features_enigneering_component_args,
        eod_balance_training=core_fair_eod_balance_training_task.outputs[
            "eod_balance_training_output"
        ],
        incidents=create_incidents_task.outputs["incidents_output"],
    )

    # Feature Selection
    feature_selection_task = feature_selection_component(
        **feature_selection_component_args,
        training_features_input=features_engineering_task.outputs[
            "training_features_output"
        ],
        core_training_input=core_fair_eod_balance_training_task.outputs[
            "core_training_output"
        ],
    )

    # Hyperparameter Tuning
    hyperparameter_tuning_task = hyperparameter_tuning_component(
        **hyperparameter_tuning_component_args,
        drivers_input=feature_selection_task.outputs["drivers_output"],
        training_drivers_input=feature_selection_task.outputs[
            "training_drivers_output"
        ],
        core_training_input=core_fair_eod_balance_training_task.outputs[
            "core_training_output"
        ],
    )

    # Condition Component: If model performance is over PROMOTION_PERFORMANCE trigger component:
    condition_deployment = (
        hyperparameter_tuning_task.outputs[  # pylint: disable=no-member
            "auc_validation"
        ]
        >= PROMOTION_PERFORMANCE
    )
    with dsl.If(
        name="Model Deployment Check",
        # TODO: component return validation auc but should use better approach
        condition=condition_deployment,
    ):

        # Model Upload -
        # https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.14.1/api/v1/model.html#v1.model.ModelUploadOp
        model_upload_task = modelregistry_component(
            **model_registry_component_args,
            best_model_input=hyperparameter_tuning_task.outputs[  # pylint: disable=no-member
                "best_model_output"
            ],
        )

    # Create Drivers
    # create_drivers_task = None

    # Get Predictions
    # get_predictions_task = None

    # Set dependencies sequence
    preprocess_eod_balance_task.after(create_eod_balance_task)
    aggregation_eod_balance_task.after(preprocess_eod_balance_task)
    create_incidents_task.after(preprocess_eod_balance_task)
    feature_selection_task.after(features_engineering_task)
    hyperparameter_tuning_task.after(  # pylint: disable=no-member
        feature_selection_task
    )
    model_upload_task.after(hyperparameter_tuning_task)

    # TODO: Implement the following components
    # create_drivers_task.after(preprocess_eod_balance_task)

    batch_prediction_task = None


# --------------------------------------------------------


# Version
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
time_based_version = f"_v_{timestamp}"
VERSION = time_based_version
# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=f"pipelines/{PIPELINE_NAME}/registry/{PIPELINE_NAME}_pipeline{VERSION}.json",
)
