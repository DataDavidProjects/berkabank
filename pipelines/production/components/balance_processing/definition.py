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
def process_eod_balance_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: List[str],
    output_files: List[str],
    column_mapping: Dict[str, str],
    transaction_usage_flag: int,
    seniority_account_flag: int,
    target_balance: int,
    incident_duration_days: int,
    compute_target: bool,
    eod_balance: dsl.Input[dsl.Dataset],
    eod_balance_preprocessed_output: dsl.Output[dsl.Dataset],
):
    from src.processing.balance import EODBalancePreprocessing
    from src.data.loader import DataLoader
    import pandas as pd

    # Load the data
    data_loader = DataLoader()
    data = data_loader.load_bucket(
        path=input_bucket_path,
        files=input_files,
    )

    data["eod_balance"] = pd.read_csv(eod_balance.path)

    # Process end-of-day balance
    eod_balance_processor = EODBalancePreprocessing(
        transaction_usage_flag=transaction_usage_flag,
        seniority_account_flag=seniority_account_flag,
        target_balance=target_balance,
        incident_duration_days=incident_duration_days,
        compute_target=compute_target,
        eod_balance=data["eod_balance"],
        column_mapping=column_mapping,
    )
    eod_balance_preprocessed = eod_balance_processor.run()

    # Save the data
    eod_balance_preprocessed_file_name = output_files[0]
    eod_balance_preprocessed.to_csv(
        f"{output_bucket_path}{eod_balance_preprocessed_file_name}.csv",
        index=False,
    )

    # Metadata
    # Add metadata to the output artifact
    eod_balance_preprocessed_output.metadata["num_rows"] = len(eod_balance_preprocessed)
    eod_balance_preprocessed_output.metadata["num_columns"] = len(
        eod_balance_preprocessed.columns
    )
    eod_balance_preprocessed_output.metadata["column_names"] = list(
        eod_balance_preprocessed.columns
    )


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    process_eod_balance_component,
    COMPONENT_FILE,
)
