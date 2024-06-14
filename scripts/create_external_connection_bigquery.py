import os
import sys
import pandas as pd
import os

sys.path.append(".")

from pipelines.production.utils.bigquery import BigQueryConf

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")


# Create a BigQuery configuration
bucket_name = BUCKET_NAME
dataset_id = BUCKET_NAME
bq_conf = BigQueryConf()
# bq_conf.create_dataset(dataset_id)

# Create a Cloud Storage connection
folder = "production/data/01_raw"
uris = bq_conf.get_uris(bucket_name, folder)
bq_conf.create_cloudstorage_connection(uris, dataset_id)


# bq_conf.create_dataset(dataset_id)

# # Create a Cloud Storage connection
folders = [
    "production/data/01_raw",
    "production/data/02_elaboration",
    "production/data/03_primary",
    "production/data/04_processing",
    "production/data/05_features",
    "production/data/06_scoring",
    # "production/data/07_output",
    "production/data/08_reporting",
]
# folder = "production/data/01_raw"
for folder in folders:
    uris = bq_conf.get_uris(bucket_name, folder)
    bq_conf.create_cloudstorage_connection(uris, dataset_id)

# Delete all tables in the dataset
# bq_conf.delete_tables_dataset(dataset_id)
