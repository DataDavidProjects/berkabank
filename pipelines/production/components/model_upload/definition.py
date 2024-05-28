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
def modelregistry_component(
    model_name: str,
    serving_naive_runtime_container_image: str,
    is_default_version: bool,
    version_aliases: List[str],
    version_description: str,
    serving_container_ports: int,
    serving_container_health_route: str,
    serving_container_predict_route: str,
    labels: Dict[str, str],
    project_id: str,
    region: str,
    best_model_input: dsl.Input[dsl.Artifact],
    model_endpoint_output: dsl.Output[dsl.Artifact],
):
    from google.cloud import aiplatform as vertex_ai

    # Init VertexAI
    vertex_ai.init(project=project_id, location=region)

    # Define common parameters
    common_params = {
        "serving_container_image_uri": serving_naive_runtime_container_image,
        "serving_container_health_route": serving_container_health_route,
        "serving_container_predict_route": serving_container_predict_route,
        "serving_container_ports": [serving_container_ports],
        "labels": labels,
    }

    # Check if the model already exists
    models = vertex_ai.Model.list(filter=f"display_name={model_name}")
    if models:
        parent_model = models[0].resource_name
        # If the model exists, upload a new version
        upload_config = vertex_ai.Model.upload(
            parent_model=parent_model,
            display_name=model_name,
            is_default_version=is_default_version,
            version_aliases=version_aliases,
            version_description=version_description,
            **common_params,
        )
    else:
        # Upload Model Registry
        upload_config = vertex_ai.Model.upload(
            display_name=model_name,
            is_default_version=is_default_version,
            version_aliases=version_aliases,
            version_description=version_description,
            **common_params,
        )

    # Create Endpoint
    model_endpoint = f"{model_name}_endpoint"
    # Fetch existing endpoint if exist
    endpoints = vertex_ai.Endpoint.list(
        filter='display_name="{}"'.format(model_endpoint),
        order_by="create_time desc",
        project=project_id,
        location=region,
    )
    # If endpoint exists take most recent otherwise create endpoint
    if len(endpoints) > 0:
        # UnDeploy Old Model
        endpoint = endpoints[0]  # most recently created
        deployed_model_id = endpoint.gca_resource.deployed_models[0].id
        endpoint.undeploy(deployed_model_id)

    else:
        # Create Endpoint
        endpoint = vertex_ai.Endpoint.create(
            display_name=model_endpoint, project=project_id, location=region
        )

    # Make Champion Model and Deploy to Endpoint
    deployed_model_display_name = f"{model_name}_champion"
    endpoint_config = upload_config.deploy(
        machine_type="n1-standard-4",
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name=deployed_model_display_name,
    )

    # Metadata
    model_endpoint_output.metadata = {
        "model_name": model_name,
        "deployed_model_display_name": deployed_model_display_name,
    }


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    modelregistry_component,
    COMPONENT_FILE,
)
