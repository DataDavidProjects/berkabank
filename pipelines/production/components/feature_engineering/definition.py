import os
from typing import Dict, List, Any
from kfp import dsl
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kfp import compiler


load_dotenv(find_dotenv())


# Define the details component
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
REPOSITORY = os.environ.get("BUCKET_NAME")  # Match the Bucket name on Artifact Registry

PIPELINE_NAME = (
    Path(__file__).resolve().parents[2].name
)  # Match the directory name of pipeline
COMPONENT_NAME = os.path.basename(os.path.dirname(__file__))  # Match the directory name
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{PIPELINE_NAME}:latest"


@dsl.component(
    base_image=BASE_IMAGE,
)
def feature_engineering_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: Dict[str, str],
    output_files: Dict[str, str],
    column_mapping: Dict[str, Any],
    aggregations: dict,
    eod_balance_training: dsl.Input[dsl.Dataset],
    incidents: dsl.Input[dsl.Dataset],
    training_features_output: dsl.Output[dsl.Dataset],
):
    from src.features.engineering import (
        FeatureEngineering,
        IncidentFeatures,
        EODBFeatures,
        PrimaryFeatures,
        DerivedFeatures,
    )
    import pandas as pd

    # Load the data
    data = {
        "eod_balance_training": pd.read_csv(
            f"{input_bucket_path}04_processing/{input_files['eod_balance_training']}.csv"
        ),
        "accounts": pd.read_csv(
            f"{input_bucket_path}03_primary/{input_files['accounts']}.csv"
        ),
        "incidents": pd.read_csv(
            f"{input_bucket_path}03_primary/{input_files['incidents']}.csv"
        ),
    }

    # Configure each class with specific parameters
    incident_features = IncidentFeatures(data["incidents"], column_mapping["incidents"])
    eodb_features = EODBFeatures(
        data["eod_balance_training"],
        column_mapping["eod_balance_training"],
        aggregations=aggregations,
    )
    primary_features = PrimaryFeatures(
        incident_features,
        eodb_features,
    )
    derived_features = DerivedFeatures(primary_features)

    # Run the FeatureEngineering pipeline
    feature_engineering = FeatureEngineering(primary_features, derived_features)
    training_features = feature_engineering.run()

    # TODO: replace with FE logic
    training_features = (
        data["eod_balance_training"]
        .loc[:, ["account_id", "n_transactions", "days_since_account_creation"]]
        .groupby("account_id")
        .sum()
    )

    # Save the data
    features_file_name = output_files["training_features"]
    training_features.to_csv(
        f"{output_bucket_path}{features_file_name}.csv",
        index=True,
    )

    # Metadata
    training_features_output.metadata["num_rows"] = len(training_features)
    training_features_output.metadata["num_columns"] = len(training_features.columns)
    training_features_output.metadata["column_names"] = list(training_features.columns)


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    feature_engineering_component,
    COMPONENT_FILE,
)
