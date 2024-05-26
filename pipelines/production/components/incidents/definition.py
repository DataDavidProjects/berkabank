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
def incidents_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: List[str],
    output_files: List[str],
    column_mapping: Dict[str, str],
    incident_duration_days: int,
    off_set_period_days: int,
    eod_balance_preprocessed: dsl.Input[dsl.Dataset],
    incidents_output: dsl.Output[dsl.Dataset],
):
    from src.processing.incidents import IncidentsBuilder
    from src.data.loader import DataLoader
    import pandas as pd

    # Load the data
    data_loader = DataLoader()
    data = data_loader.load_bucket(
        path=input_bucket_path,
        files=input_files,
    )

    # Process end-of-day balance
    incidents_processor = IncidentsBuilder(
        eod_balance=data["eod_balance_preprocessed"],
        accounts=data["accounts"],
        incident_duration_days=incident_duration_days,
        off_set_period_days=off_set_period_days,
        column_mapping=column_mapping,
    )
    incidents_preprocessed = incidents_processor.run()

    # Save the data
    incidents_preprocessed_file_name = output_files[0]
    incidents_preprocessed.to_csv(
        f"{output_bucket_path}{incidents_preprocessed_file_name}.csv",
        index=False,
    )

    # Metadata
    # Add metadata to the output artifact
    incidents_output.metadata["num_rows"] = len(incidents_preprocessed)
    incidents_output.metadata["num_columns"] = len(incidents_preprocessed.columns)
    incidents_output.metadata["column_names"] = list(incidents_preprocessed.columns)


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    incidents_component,
    COMPONENT_FILE,
)
