import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from google.auth import default, exceptions
from google.cloud import aiplatform

load_dotenv(find_dotenv())


def vertex_authenticate():
    """Authenticate with Google Cloud SDK."""
    # Authenticate with Google Cloud SDK
    try:
        credentials, _ = default()
        aiplatform.init(
            project=os.environ.get("PROJECT_ID"),
            credentials=credentials,
            location=os.environ.get("REGION"),
            staging_bucket=f"gs://{os.environ.get('BUCKET_NAME')}",
        )
        return aiplatform
        # print("Authenticated with Google Cloud SDK successfully.")
        # print(f"Project ID: {project} \nRegion: {self.config.get('region')}")
    except exceptions.DefaultCredentialsError:
        print("Please authenticate with Google Cloud SDK.")
