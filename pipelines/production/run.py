from google.cloud.aiplatform import pipeline_jobs
from pathlib import Path
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
import os
from utils.bigquery import BigQueryConf


# Load environment variables from .env file
load_dotenv(find_dotenv())


# Define the pipeline
PIPELINE_NAME = Path(__file__).resolve().parents[0].name


PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

# Version - up to minute to match pipeline definition
timestamp_version = datetime.now().strftime("%Y%m%d%H%M")
time_based_version = f"_v_{timestamp_version}"
VERSION = time_based_version


start_pipeline = pipeline_jobs.PipelineJob(
    display_name=PIPELINE_NAME,
    template_path=f"pipelines/{PIPELINE_NAME}/registry/{PIPELINE_NAME}_pipeline{VERSION}.json",
    enable_caching=False,
    location=REGION,
    project=PROJECT_ID,
    job_id=f"{PIPELINE_NAME}-{TIMESTAMP}",
)


# Run the pipeline
# Note : Update Containter if new changes are made in src
start_pipeline.run()


BQsetup = False
# Create a BigQuery Connections for new generated tables, if exists
bucket_name = os.environ.get("BUCKET_NAME")
dataset_id = os.environ.get("BUCKET_NAME")
bq_conf = BigQueryConf()

# Create Dataset for the first time.
if BQsetup:
    bq_conf.create_dataset(dataset_id)

# Create a Cloud Storage connection
folders = [
    "production/data/01_raw",
    "production/data/02_elaboration",
    "production/data/03_primary",
    "production/data/04_processing",
    "production/data/05_features",
    "production/data/06_scoring",
    "production/data/07_output",
    "production/data/08_reporting",
]
# folder = "production/data/01_raw"
for folder in folders:
    uris = bq_conf.get_uris(bucket_name, folder)
    bq_conf.create_cloudstorage_connection(uris, dataset_id)
