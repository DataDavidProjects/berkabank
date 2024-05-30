import os
from typing import Dict, List, Any, Union
from kfp import dsl
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kfp import compiler
from google.cloud import aiplatform

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
    region: str,
    model_resource_name: str,
    job_display_name: str,
    gcs_source: str,
    gcs_destination: str,
    instances_format: str,
    machine_type: str,
    accelerator_count: int,
    starting_replica_count: int,
    max_replica_count: int,
    sync: bool,
    batch_prediction_job: dsl.Output[dsl.Artifact],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    def get_model_by_display_name(display_name, verbose=False):
        client = aiplatform.gapic.ModelServiceClient(
            client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
        )
        parent = f"projects/{project_id}/locations/{region}"
        response = client.list_models(parent=parent)

        for model in response:
            if model.display_name == display_name:
                if verbose:
                    print(f"Model {display_name} found.")
                    print(f"Model details:\n {model}")
                return model
        else:
            print(f"Model {display_name} not found.")

    model_id = get_model_by_display_name(model_resource_name).name
    model_container = aiplatform.Model(model_id)

    batch_prediction_job = model_container.batch_predict(
        job_display_name=job_display_name,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination,
        instances_format=instances_format,
        machine_type=machine_type,
        accelerator_count=accelerator_count,
        accelerator_type=None,
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
compiler.Compiler().compile(
    batch_prediction_component,
    COMPONENT_FILE,
)
