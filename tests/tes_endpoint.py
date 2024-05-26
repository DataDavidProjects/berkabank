import json
from google.cloud import aiplatform
import pandas as pd


# Define the project, location, and endpoint ID
project_id = "opencreator-1699308232742"
project_number = "1036389498447"
location = "europe-west6"
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
