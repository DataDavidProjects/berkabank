import sys

sys.path.append(".")
from utils.pipeline import LazyPipe
from google.cloud import storage
from google.cloud import aiplatform as vertex_ai
import os
import joblib
import pandas as pd
from google.cloud import aiplatform

MODEL_NAME = "berkamodel"
PIPELINE_NAME = "production"
# Authenticate with Google Cloud SDK for Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
REPOSITORY = os.environ.get("BUCKET_NAME")  # Match the Bucket name on Artifact Registry
# Define the pipeline name
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{PIPELINE_NAME}:latest"


# -------------  Build Container for LazyPipe  -------------
# Define the setup for the pipeline
setup = False
build = True
# Define the pipelines to run
pipes = ["production"]
for pipe in pipes:
    print(f"Running {pipe} pipeline...")
    lazypipe = LazyPipe(pipe=pipe)
    lazypipe.create_container()


# -------------  Upload Model Registy  -------------

# Init VertexAI
vertex_ai.init(project=PROJECT_ID, location=REGION)

upload_config = vertex_ai.Model.upload(
    display_name=MODEL_NAME,
    is_default_version=False,
    version_aliases=["experimental", "challenger", "custom-training", "random-forest"],
    version_description="A classifier",
    serving_container_image_uri=BASE_IMAGE,
    serving_container_health_route="/health",
    serving_container_predict_route="/predict",
    serving_container_ports=[8080],
    labels={"created_by": "davide", "team": "badass"},
)

# -------------  Deploy Model to Endpoint  -------------
model_endpoint = f"{MODEL_NAME}_endpoint"
# Fetch existing endpoint if exist
endpoints = vertex_ai.Endpoint.list(
    filter='display_name="{}"'.format(model_endpoint),
    order_by="create_time desc",
    project=PROJECT_ID,
    location=REGION,
)
# If endpoint exists take most recent otherwise create endpoint
print(endpoints)
if len(endpoints) > 0:
    endpoint = endpoints[0]  # most recently created
else:
    # Create Endpoint
    endpoint = vertex_ai.Endpoint.create(
        display_name=model_endpoint, project=PROJECT_ID, location=REGION
    )


# Make Champion Model and Deploy to Endpoint
deployed_model_display_name = f"{MODEL_NAME}_champion"
endpoint_config = upload_config.deploy(
    deployed_model_display_name=deployed_model_display_name,
    endpoint=endpoint,
    min_replica_count=1,
    max_replica_count=1,
)

# ------------- Test Endpoint  -------------
# Define the project, location, and endpoint ID
project_id = os.environ.get("PROJECT_ID")
project_number = os.environ.get("PROJECT_NUMBER")
location = os.environ.get("REGION")
endpoint_id = "1986360114554077184"
# Authenticate the client
aiplatform.init(project=project_id, location=location)
# Initialize the AI Platform endpoint
endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{project_number}/locations/{location}/endpoints/{endpoint_id}"
)
# Create the instances
instances = [
    {"n_transactions": 10},
    {"n_transactions": 20},
    {"n_transactions": 30},
]
# Make the prediction
prediction = endpoint.predict(instances=instances)
print(pd.DataFrame(prediction.predictions))
