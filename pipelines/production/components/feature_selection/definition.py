import os
from typing import Dict, List, Any, Union
from kfp import dsl
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kfp import compiler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator


# Load ENV
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
def feature_selection_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: Dict[str, str],
    output_files: Dict[str, str],
    params: Dict[str, Any],
    step: float,
    cv: int,
    scoring: str,
    n_jobs: int,
    standard_error_threshold: float,
    return_type: str,
    num_features: str,
    core_training_input: dsl.Input[dsl.Dataset],
    training_features_input: dsl.Input[dsl.Dataset],
    training_drivers_output: dsl.Output[dsl.Dataset],
    drivers_output: dsl.Output[dsl.Artifact],
):
    from src.features.selection import FeatureEliminationShap
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import json
    from google.cloud import storage

    # Load the data
    data = {
        "training_features": pd.read_csv(
            f"{input_bucket_path}05_features/{input_files['training_features']}.csv"
        ),
        "core_training": pd.read_csv(
            f"{input_bucket_path}04_processing/{input_files['core_training']}.csv"
        ),
    }

    # Model GridSearch
    model = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions=params,
    )

    # Feature Elimination with Shap Values and GridSearch
    features_processor = FeatureEliminationShap(
        model=model,
        step=step,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        standard_error_threshold=standard_error_threshold,
        return_type=return_type,
        num_features=num_features,
    )
    y = data["core_training"].set_index("account_id")["target"]
    X = data["training_features"].set_index("account_id")
    drivers = features_processor.run(X=X, y=y)
    # Filter data with drivers
    # TODO update with actual logic FE
    features_primary = X.loc[:, drivers]

    # Save the data
    features_preprocessed_file_name = output_files["training_drivers"]
    # Select drivers
    features_primary.to_csv(
        f"{output_bucket_path}{features_preprocessed_file_name}.csv",
        index=True,
    )
    # Save Drivers to CS
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("berkabank")
    blob = bucket.blob("production/artifacts/model/drivers.json")
    blob.upload_from_string(json.dumps(drivers))

    # Metadata
    training_drivers_output.metadata["num_rows"] = len(features_primary)
    training_drivers_output.metadata["num_columns"] = len(features_primary.columns)
    training_drivers_output.metadata["column_names"] = list(features_primary.columns)

    # Driver Metadata in custom artifact
    drivers_output.metadata["drivers"] = drivers


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    feature_selection_component,
    COMPONENT_FILE,
)
