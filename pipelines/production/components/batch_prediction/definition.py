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
def batch_prediction_component(
    project_id: str,
    location: str,
    model_resource_name: str,
    job_display_name: str,
    gcs_source: str,
    gcs_destination: str,
    instances_format: str,
    machine_type: str,
    accelerator_count: int,
    accelerator_type: str,
    starting_replica_count: int,
    max_replica_count: int,
    sync: bool,
    batch_prediction_job: dsl.Output[dsl.Artifact],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    my_model = aiplatform.Model(model_resource_name)

    batch_prediction_job = my_model.batch_predict(
        job_display_name=job_display_name,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination,
        instances_format=instances_format,
        machine_type=machine_type,
        accelerator_count=accelerator_count,
        accelerator_type=accelerator_type,
        starting_replica_count=starting_replica_count,
        max_replica_count=max_replica_count,
        sync=sync,
    )

    batch_prediction_job.wait()

    batch_prediction_job.metadata = {
        "display_name": batch_prediction_job.display_name,
        "resource_name": batch_prediction_job.resource_name,
        "state": batch_prediction_job.state,
    }


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/batch_prediction.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    batch_prediction_component,
    COMPONENT_FILE,
)
