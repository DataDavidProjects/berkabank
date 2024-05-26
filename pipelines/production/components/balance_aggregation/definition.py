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
def eod_balance_aggregation_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: List[str],
    output_files: List[str],
    column_mapping: Dict[str, str],
    incident_duration_days: int,
    off_set_period_days: int,
    eod_balance_preprocessed: dsl.Input[dsl.Dataset],
    eod_balance_aggregation_output: dsl.Output[dsl.Dataset],
):

    from src.processing.balance import EodBalanceAggregation
    from src.data.loader import DataLoader
    import pandas as pd

    # Load the data
    data_loader = DataLoader()
    data = data_loader.load_bucket(
        path=input_bucket_path,
        files=input_files,
    )

    # Aggregate the data
    aggregation = EodBalanceAggregation(
        data["eod_balance_preprocessed"],
        data["accounts"],
        incident_duration_days,
        off_set_period_days,
        column_mapping=column_mapping,
    )

    fair_eod_balance_period = aggregation.run()

    # Save the data
    fair_eod_balance_period_file_name = output_files[0]
    fair_eod_balance_period.to_csv(
        f"{output_bucket_path}{fair_eod_balance_period_file_name}.csv",
        index=False,
    )

    # Metadata
    # Add metadata to the output artifact
    eod_balance_aggregation_output.metadata["num_rows"] = len(fair_eod_balance_period)
    eod_balance_aggregation_output.metadata["num_columns"] = len(
        fair_eod_balance_period.columns
    )
    eod_balance_aggregation_output.metadata["column_names"] = list(
        fair_eod_balance_period.columns
    )


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    eod_balance_aggregation_component,
    COMPONENT_FILE,
)
