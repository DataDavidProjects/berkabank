import os
from dataclasses import dataclass
from google.api_core.exceptions import AlreadyExists, Conflict
from google.cloud import bigquery
from google.cloud import artifactregistry, storage
from dotenv import load_dotenv, find_dotenv


# Load ENV
load_dotenv(find_dotenv())

# Define the details component
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")


@dataclass
class CloudStorageConfig:
    """Cloud Storage configuration.

    Attributes:
        client (storage.Client): Google Cloud Storage client.
        config (dict): Configuration dictionary.
        bucket (storage.Bucket): Google Cloud Storage bucket.
    """

    client: storage.Client
    config: dict
    bucket: storage.Bucket = None

    def create_bucket(self) -> storage.Bucket:
        """Create a Cloud Storage bucket.

        Args:
            bucket_name (str): name of the bucket to create

        Returns:
            storage.Bucket: created bucket
        """
        try:
            # Create a new bucket
            bucket = self.client.create_bucket(
                bucket_or_name=self.config["bucket_name"],
                location=self.config["region"],
            )

            # Add the service account as an object admin if provided
            if self.config["service_account"]:
                policy = bucket.get_iam_policy(requested_policy_version=3)
                policy.bindings.append(
                    {
                        "role": "roles/storage.objectAdmin",
                        "members": [f'serviceAccount:{self.config["service_account"]}'],
                    }
                )
                bucket.set_iam_policy(policy)
            # Set the bucket property
            self.bucket = bucket

        except Conflict:
            print("The bucket already exists. Skipping creation...")
            self.bucket = self.client.get_bucket(self.config["bucket_name"])

        return self

    def template_directories(self) -> None:
        """Create multiple directories in the bucket.

        Args:
            bucket (storage.Bucket): bucket to create directories in
        """
        print(f"Creating directories in bucket: {self.config['directories']}")
        root = self.config["directories"]
        print(f"Creating directory: {root}")
        subdirectories = [
            "run",
            "data/01_raw",
            "data/02_elaboration",
            "data/03_primary",
            "data/04_processing",
            "data/05_features",
            "data/06_scoring",
            "data/07_output",
            "data/08_reporting",
            "artifacts/model",
            "artifacts/pipeline",
            "artifacts/transformer",
        ]
        for subdirectory in subdirectories:
            blob = self.bucket.blob(f"{root}/{subdirectory}/")
            if not blob.exists():
                print(f"Creating subdirectory: {root}/{subdirectory}")
                blob.upload_from_string("")
            else:
                print(
                    f"Subdirectory {root}/{subdirectory} already exists. Skipping creation."
                )


@dataclass
class ArtifactRegistryConfig:
    """Artifact Registry configuration.

    Attributes:
        client (artifactregistry.ArtifactRegistryClient): Google Cloud Artifact Registry client.
        config (dict): Configuration dictionary.
    """

    client: artifactregistry.ArtifactRegistryClient
    config: dict

    def authenticate(self):
        os.system(f"gcloud auth application-default login")

    def create_repository(self) -> str:
        """Creates a new repository in the GCP Artifact Registry based on docker image.

        Args:
            project_id (str): project ID or project number of the Cloud project.
            location (str): location you want to use. For a list of locations.
            repository_id (str): the name of the repository.

        Returns:
            The name of the created repository.
        """
        # os.system(f"gcloud auth configure-docker {self.config.get('region')}-docker.pkg.dev")
        # REQ: gcloud auth configure-docker europe-west6-docker.pkg.dev
        try:
            parent = f"projects/{self.config.get('project_id')}/locations/{self.config.get('region')}"
            repository = artifactregistry.Repository()
            repository.format = artifactregistry.Repository.Format.DOCKER
            response = self.client.create_repository(
                request={
                    "parent": parent,
                    # Repository_id match the bucket name for consistency
                    "repository_id": self.config.get("bucket_name"),
                    "repository": repository,
                }
            )
            print("Created Repository in Artifact Registry")

            return response

        except AlreadyExists:
            print("The repository already exists. Skipping creation...")
            return None


@dataclass
class DockerConfig:
    """Docker configuration.

    Attributes:
        config (dict): Configuration dictionary.
    """

    config: dict

    def build_image(self):
        """Builds a docker image based on the provided configuration."""
        # Build the docker image
        cmd = f"docker build --build-arg PIPELINE_NAME={self.config.get('image_name')} --build-arg PROJECT_ID={PROJECT_ID} --build-arg REGION={REGION} --build-arg BUCKET_NAME={BUCKET_NAME}  -t {self.config.get('image_name')} -f {self.config.get('dockerfile_path')} ."
        print(f"\nRunning Command:\n{cmd}\n")
        os.system(cmd)
        return self

    def tag_image(self):
        """Tags a Docker image based on the provided configuration."""
        # Tag the Docker image
        cmd = f"docker tag {self.config.get('image_name')} {self.config.get('region')}-docker.pkg.dev/{self.config.get('project_id')}/{self.config.get('repository_id')}/{self.config.get('image_name')}:{self.config.get('image_tag')}"
        print(f"\nRunning Command:\n{cmd}\n")
        os.system(cmd)
        return self

    def push_image(self):
        """Push a docker image based on the provided configuration on GCP."""
        os.system(
            f"gcloud auth configure-docker {self.config.get('region')}-docker.pkg.dev"
        )
        # Push the docker image
        base_image = f"{self.config.get('region')}-docker.pkg.dev/{self.config.get('project_id')}/{self.config.get('repository_id')}/{self.config.get('image_name')}:{self.config.get('image_tag')}"
        cmd = f"docker push {base_image}"
        print(f"\nRunning Command:\n{cmd}\n")
        os.system(cmd)
        return self

    def create_container(self):
        """Creates and Push container based on the provided configuration."""
        self.build_image().tag_image().push_image()


@dataclass
class ProjectConfig:
    """Project configuration.
    Attributes:
        config (dict): Configuration dictionary.
    """

    config: dict

    def enable_apis(self):
        """Activates the specified Google Cloud APIs."""
        cmd = """
        gcloud services enable compute.googleapis.com \
                           containerregistry.googleapis.com  \
                           aiplatform.googleapis.com  \
                           cloudbuild.googleapis.com \
                           cloudfunctions.googleapis.com \
                           bigquery.googleapis.com
        """
        print(f"\nRunning Command:\n{cmd}\n")
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise Exception("Failed to run command. Is 'gcloud' installed?")
