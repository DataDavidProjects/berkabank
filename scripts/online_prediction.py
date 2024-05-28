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
