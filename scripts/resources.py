import os

from dotenv import find_dotenv, load_dotenv
from google.cloud import artifactregistry, storage

from utils.project import ArtifactRegistryConfig, CloudStorageConfig, ProjectConfig


pipes = ["training", "deployment"]
# Set up Project Config
load_dotenv(find_dotenv())
project_config = {
    # Environment Variables
    "project_id": os.environ.get("PROJECT_ID"),
    "project_number": os.environ.get("PROJECT_NUMBER"),
    "region": os.environ.get("REGION"),
    "service_account": os.environ.get("SERVICE_ACCOUNT"),
    "bucket_name": os.environ.get("BUCKET_NAME"),
    # Note: match directory names with pipeline names in pipelines directory
    "directories": pipes,
}

# Initialize Project Config
project_config = ProjectConfig(config=project_config)

# Enable APIs
project_config.enable_apis()


# Initialzie Cloud Storage Client
storage_client = storage.Client()

# Initialize Cloud Storage Config
cloud_storage_config = CloudStorageConfig(client=storage_client, config=project_config)
# Create a bucket and template directories
cloud_storage_config.create_bucket().template_directories()


# Initialize Artifact Registry Client
artifactregistry_client = artifactregistry.ArtifactRegistryClient()

# Initialize Artifact Registry Config
artifactregistry_config = ArtifactRegistryConfig(
    client=artifactregistry_client, config=project_config
)

# Create a repository in Artifact Registry for Docker Image
artifactregistry_config.create_repository()
