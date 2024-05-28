import os
from typing import Dict, List, Any, Union
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
def create_model_drivers(
    input_bucket_path: str,
    input_files: List[str],
    column_mapping: Dict[str, str],
    drives_path: str,
    eod_preprocess_input: dsl.Input[dsl.Dataset],
    driver_output: dsl.Output[dsl.Dataset],
):

    from src.features.drivers import EODBDrivers
    from src.data.loader import DataLoader
    import pandas as pd

    # Load the data
    data_loader = DataLoader()
    data = data_loader.load_bucket(
        path=input_bucket_path,
        files=input_files,
    )

    # Build model drivers
    drivers_builder = EODBDrivers(
        eod_balance_preprocessed=data["eod_balance_preprocessed"],
        column_mapping=column_mapping,
    )

    drivers = drivers_builder.run()

    bucket_name, file_name = drives_path.replace("gs://", "").split("/", 1)
    driver_list = data_loader.load_json_from_bucket(bucket_name, file_name)

    # Assert the drivers are in the columns of the dataset
    assert set(driver_list).issubset(set(drivers.columns))


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    create_model_drivers,
    COMPONENT_FILE,
)
