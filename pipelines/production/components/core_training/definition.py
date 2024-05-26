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
def core_fair_eod_balance_training_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: Dict[str, str],
    output_files: Dict[str, str],
    column_mapping: Dict[str, str],
    eod_balance_aggregation: dsl.Input[dsl.Dataset],
    eod_balance_training_output: dsl.Output[dsl.Dataset],
    core_training_output: dsl.Output[dsl.Dataset],
):
    from src.data.core import CoreBuilderTraining
    import pandas as pd

    # Load the data
    data = {
        "eod_balance_aggregation": pd.read_csv(
            f"{input_bucket_path}04_processing/{input_files['eod_balance_aggregation']}.csv"
        ),
        "accounts": pd.read_csv(
            f"{input_bucket_path}03_primary/{input_files['accounts']}.csv"
        ),
        "incidents": pd.read_csv(
            f"{input_bucket_path}03_primary/{input_files['incidents']}.csv"
        ),
    }

    # Process core
    core_processor = CoreBuilderTraining(
        accounts=data["accounts"],
        eod_balance_agg=data["eod_balance_aggregation"],
        incidents=data["incidents"],
        column_mapping=column_mapping,
    )
    core_preprocessed = core_processor.run()
    print(f"core_preprocessed:\n{core_preprocessed}")
    eod_balance_training = core_preprocessed["eod_balance_training"]
    core_training = core_preprocessed["training_core"]

    # Save the data
    eod_balance_training_file_name = output_files["eod_balance_training"]
    eod_balance_training.to_csv(
        f"{output_bucket_path}{eod_balance_training_file_name}.csv",
        index=False,
    )
    core_training_file_name = output_files["core_training"]
    core_training.to_csv(
        f"{output_bucket_path}{core_training_file_name}.csv",
        index=False,
    )

    # Metadata
    # Add metadata to the output artifact
    eod_balance_training_output.metadata["num_rows"] = len(eod_balance_training)
    eod_balance_training_output.metadata["num_columns"] = len(
        eod_balance_training.columns
    )
    eod_balance_training_output.metadata["column_names"] = list(
        eod_balance_training.columns
    )
    core_training_output.metadata["num_rows"] = len(core_training)
    core_training_output.metadata["num_columns"] = len(core_training.columns)
    core_training_output.metadata["column_names"] = list(core_training.columns)


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    core_fair_eod_balance_training_component,
    COMPONENT_FILE,
)
