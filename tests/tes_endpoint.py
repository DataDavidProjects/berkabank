import json
from google.cloud import aiplatform
import pandas as pd
import os


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
print(prediction)
