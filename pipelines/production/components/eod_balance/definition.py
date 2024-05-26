import os
from typing import Dict, List, Any, NamedTuple
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
def create_eod_balance_component(
    column_mapping: Dict[str, str],
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: List[str],
    output_files: List[str],
    eod_balance_output: dsl.Output[dsl.Dataset],
):

    from src.data.balance import EodBalanceBuilder
    from src.data.loader import DataLoader

    # Load the data
    data_loader = DataLoader()
    data = data_loader.load_bucket(
        path=input_bucket_path,
        files=input_files,
    )

    # Create end-of-day balance
    eod_balance_builder = EodBalanceBuilder(
        transactions=data["transactions"],
        accounts=data["accounts"],
        column_mapping=column_mapping,
    )
    eod_balance = eod_balance_builder.run()

    # Save the data to Bucket
    eod_balance_file_name = output_files[0]
    eod_balance.to_csv(
        f"{output_bucket_path}{eod_balance_file_name}.csv",
        index=False,
    )
    # Write the output to a file in pipeline path to be used in the next component
    eod_balance.to_csv(
        eod_balance_output.path,
        index=False,
    )

    # Metadata
    # Add metadata to the output artifact
    eod_balance_output.metadata["num_rows"] = len(eod_balance)
    eod_balance_output.metadata["num_columns"] = len(eod_balance.columns)
    eod_balance_output.metadata["column_names"] = list(eod_balance.columns)


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    create_eod_balance_component,
    COMPONENT_FILE,
)
