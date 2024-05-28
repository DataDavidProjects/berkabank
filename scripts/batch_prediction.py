# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
from google.cloud import storage
from google.cloud import aiplatform
import os

# Constants
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_NAME = "berkamodel"
PIPELINE_NAME = "production"
MACHINE_TYPE = "n1-standard-2"
FILE_SCORING_NAME = "training_drivers"
MODEL_RESOURCE_NAME = f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_NAME}"
JOB_DISPLAY_NAME = f"{MODEL_NAME}_batch_prediction_job"
GCS_SOURCE = (
    f"gs://{BUCKET_NAME}/{PIPELINE_NAME}/data/05_features/{FILE_SCORING_NAME}.csv"
)
GCS_DESTINATION = f"gs://{BUCKET_NAME}/{PIPELINE_NAME}/data/07_output/"
INSTANCES_FORMAT = "csv"
MACHINE_TYPE = "n1-standard-2"
ACCELERATOR_COUNT = 0
ACCELERATOR_TYPE = None
STARTING_REPLICA_COUNT = 1
MAX_REPLICA_COUNT = 1
SYNC = False


def get_model_by_display_name(display_name, verbose=False):
    client = aiplatform.gapic.ModelServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    response = client.list_models(parent=parent)

    for model in response:
        if model.display_name == display_name:
            if verbose:
                print(f"Model {display_name} found.")
                print(f"Model details:\n {model}")
            return model

    return None


# Define the details for the batch prediction job
aiplatform.init(project=PROJECT_ID, location=REGION)

model_id = get_model_by_display_name(MODEL_NAME).name
model_container = aiplatform.Model(model_id)

batch_prediction_job = model_container.batch_predict(
    job_display_name=JOB_DISPLAY_NAME,
    gcs_source=GCS_SOURCE,
    gcs_destination_prefix=GCS_DESTINATION,
    instances_format=INSTANCES_FORMAT,
    machine_type=MACHINE_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
    accelerator_type=ACCELERATOR_TYPE,
    starting_replica_count=STARTING_REPLICA_COUNT,
    max_replica_count=MAX_REPLICA_COUNT,
    sync=SYNC,
)

batch_prediction_job.wait()
