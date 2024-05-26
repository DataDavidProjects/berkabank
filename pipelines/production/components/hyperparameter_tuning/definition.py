import os
from typing import Dict, List, Any, NamedTuple, Union
from kfp import dsl
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kfp import compiler

from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
def hyperparameter_tuning_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: Dict[str, str],
    output_files: Dict[str, str],
    params: Dict[str, Any],
    n_splits: int,
    n_iter: int,
    scoring: str,
    model_name: str,
    drivers_input: dsl.Input[dsl.Artifact],
    training_drivers_input: dsl.Input[dsl.Dataset],
    core_training_input: dsl.Input[dsl.Dataset],
    hpt_grid_output: dsl.Output[dsl.Artifact],
    hpt_report_output: dsl.Output[dsl.Dataset],
    best_model_output: dsl.Output[dsl.Model],
) -> NamedTuple("Outputs", [("auc_validation", float)]):

    from sklearn.model_selection import StratifiedShuffleSplit
    from src.model.hyperparameter import HyperparameterTuning
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, auc
    import pandas as pd
    import json
    import numpy as np
    from google.cloud import storage
    import joblib

    # Load the data
    data = {
        # Index by account_id
        "training_drivers": pd.read_csv(
            f"{input_bucket_path}05_features/{input_files['training_drivers']}.csv"
        ),
        "core_training": pd.read_csv(
            f"{input_bucket_path}04_processing/{input_files['core_training']}.csv"
        ),
    }

    # ----------------   Hyperparameter Tuning and Validation ------------------
    # Split train and validation
    X = data["training_drivers"].set_index("account_id")
    y = data["core_training"].set_index("account_id")["target"]

    # Balance cv and validation folds
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    for train_index, test_index in split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Pick the Model Estimator #TODO: serialize in input
    estimator = RandomForestClassifier()

    # Run HPT
    hp_processor = HyperparameterTuning(
        estimator=estimator,
        params=params,
        n_splits=n_splits,
        scoring=scoring,
        n_iter=n_iter,
        random_state=42,
    )
    hp_processor_output = hp_processor.run(X_train, y_train)
    report = hp_processor_output["report"]
    best = hp_processor_output["best"]

    # Assess Validation and Optimal CutOff
    best_estimator = estimator.set_params(**best["best_params"])
    best_estimator.fit(X_train, y_train)
    y_scores = estimator.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc_val = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"TEST ROC AUC: {roc_auc_val}")
    print(f"Optimal threshold: {optimal_threshold}")
    best["optimal_threshold"] = optimal_threshold
    # --------------------------------------------------------------------------

    # Save Data
    report.to_csv(f"{input_bucket_path}08_reporting/hpt_report_latest.csv")
    # Save Model to GCS
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("berkabank")
    # Save the model to a local file
    joblib.dump(best_estimator, f"{model_name}.joblib")
    blob = bucket.blob(f"production/artifacts/model/{model_name}.joblib")
    # Upload the local file to the cloud storage
    with open(f"{model_name}.joblib", "rb") as model_file:
        blob.upload_from_file(model_file)

    # MetaData
    hpt_grid_output.metadata["model_name"] = model_name
    hpt_grid_output.metadata["model_uri"] = (
        f"gs://berkabank/production/artifacts/model/{model_name}.joblib"
    )
    hpt_grid_output.metadata["best_params"] = best["best_params"]
    hpt_grid_output.metadata["best_score"] = best["best_score"]
    hpt_grid_output.metadata["optimal_threshold"] = best["optimal_threshold"]
    hpt_grid_output.metadata["auc_validation"] = roc_auc_val

    hpt_report_output.metadata["column_names"] = report.columns.tolist()
    hpt_report_output.metadata["num_rows"] = len(report)

    return (roc_auc_val,)


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    hyperparameter_tuning_component,
    COMPONENT_FILE,
)
