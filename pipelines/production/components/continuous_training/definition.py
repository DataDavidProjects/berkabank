import os
from typing import Dict, List, Any, NamedTuple, Union
from kfp import dsl
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kfp import compiler


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
def training_trigger_component(
    input_bucket_path: str,
    trigger_performance: float,
) -> NamedTuple("Outputs", [("trigger", str)]):

    import pandas as pd

    # Logic to trigger the training
    training_trigger = False
    try:
        # Check if the model is already in the report
        model_report = pd.read_csv(f"{input_bucket_path}08_reporting/model_report.csv")
        # Check if performance of latest model
        if model_report.tail(1)["roc_auc_validation"] < trigger_performance:
            # If performance is below the threshold, trigger the training
            training_trigger = True
    # If model is not in the report, trigger the training
    except FileNotFoundError:
        training_trigger = True

    return (str(training_trigger),)


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/batch_prediction.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    training_trigger_component,
    COMPONENT_FILE,
)
