# 🚀 Moving fast from MVP to MLOps - LazyPipelines

Not every team has machine learning engineers using IAC and MLOps tools. To make a small team of Data Scientist able to deliver fast and without using too many tools, i propose the use of a custom class named LazyPipelines.
The pro of using them are:

- You can ignore most of the Engineer side and focus on your src code
- The best practice of lazy pipelines are similar to the ones used in real world frameworks like Kedro or Terraform.

The limitations:

- LazyPipelines are not for all the usecases, it supposed to be a mid point between AutoML and Custom Code for supervised learning

## Naming conventions for files and directories

In order to eliminate the majority of the boilerplate work, some naming convention are introduced.
Inside each folder type:

1.  Pipeline
    There following files are mandatory:

    1. `definition.py`
    2. `Dockerfile`
    3. `run.py`
    4. `setup.py`
    5. `src`

2.  Components
    There following files are mandatory:
    1. `definition.py`

## Pipelines

A pipeline is named after the name of its directory in the `pipelines` directory.

```bash
project_root/
├── pipelines/
│   ├── deployment/ # the pipeline name is deployment
│   │   ├── components/ # a pipeline can have multiple components
│   │   │   ├── component 1/ # the component name is component 1
│   │   │   │   └── definition.py # compile the component
│   │   │   ├── component 2/
│   │   │   │   └── definition.py
│   │   ├── Dockerfile # Container image
│   │   ├── definition.py # compile the pipeline
│   │   ├── run.py # Launch the pipeline
│   │   ├── setup.py # Install the dependencies and src
│   │   └── registry/ # Compiled pipeline versions ready to be deployed
│   │   └── src/ # source code with business logic
│   │       └── ....
├── ...
└── main.py # Build and Launch all the LazyPipelines
```

## Components

A component is the building block of a pipeline and it is named after the name of its directory in the `components` directory.
Components are build using principles of Object Oriented Programming and Kubeflow function components.

Each component is built using the following logic in the file `definition.py`:

1. Init component details - fixed
2. Business Logic - configurable by `src`
3. Compile components to generate yaml file - fixed

Example of component definition:

```python
# definition.py
import os
from typing import Dict
from kfp import dsl
from pathlib import Path
from kfp import compiler


# ------------------COMPONENT DETAILS---------------------------
# Define the details component
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
REPOSITORY = os.environ.get("BUCKET_NAME")  # Match the Bucket name on Artifact Registry
# Match the directory name of pipeline
PIPELINE_NAME = (
    Path(__file__).resolve().parents[2].name
)
COMPONENT_NAME = os.path.basename(os.path.dirname(__file__))  # Match the directory name
# Docker image built and pushed to the artifact registry.
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{PIPELINE_NAME}:latest"
# --------------------------------------------------------------


# ------------------BUSINESS LOGIC----------------------------
@dsl.component(
    base_image=BASE_IMAGE,
) # Name of the function of the function component
def hyperparameter_tuning_component(custom_input:dsl.Input[dsl.Model]): # input type definition

    # Import a class from the src
    from src.features.selection import CustomClass
    # Note the imports are inside the function!
    # This is required because the component has its own environment

    # -----------------  Code Logic --------------------------------
    # Initialize the feature elimination object
    class_instance = CustomClass(custom_input=custom_input)

    # Run the feature elimination process
    reduced_features = class_instance.run()
    # -------------------------------------------------------------


# ------------------------------------------------------------------


# --------------------COMPILE COMPONENT-----------------------------
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(hyperparameter_tuning_component, COMPONENT_FILE)
# ------------------------------------------------------------------
```

Following this logic it is possible to create many custom components with minimal changes.

## SourceCode of a LazyPipeline class:

```python
import os
from dataclasses import dataclass
from dotenv import find_dotenv, load_dotenv
from google.cloud import artifactregistry, storage


# specify utils in import for import in main.py
from utils.project import (
    ArtifactRegistryConfig,
    CloudStorageConfig,
    ProjectConfig,
    DockerConfig,
)


# Load environment variables from .env file
load_dotenv(find_dotenv())


@dataclass
class LazyPipe:
    """
    End-to-end pipeline creation and execution process.

    Methods:
        enable_resources: Enable the required resources for the pipeline.
        set_up_storage: Set up the Cloud Storage bucket and template directories.
        set_up_artifact_registry: Set up the Artifact Registry repository.
        create_container: Create the Docker container.
        define_pipeline: Define the pipeline by running the definition script.
        run_pipeline: Run the pipeline by running the run script.
        magic: End-to-end pipeline creation and execution process.

    Attributes:
        pipe (str): The name of the pipeline.

    Internal Attributes:
        container_args (dict): The arguments for creating the Docker container.
        project_config (dict): The configuration for the project.
        storage_client (storage.Client): The Cloud Storage client.
        artifactregistry_client (artifactregistry.ArtifactRegistryClient): The Artifact Registry client.
        project_config (ProjectConfig): The ProjectConfig instance.
        artifactregistry_config (ArtifactRegistryConfig): The ArtifactRegistryConfig instance.
        cloud_storage_config (CloudStorageConfig): The CloudStorageConfig instance.
        artifactregistry_config (ArtifactRegistryConfig): The ArtifactRegistryConfig instance.
    """

    pipe: str

    def __post_init__(self):
        """
        Initialize the DockerConfig instance and create the Docker container.
        """
        self.container_args = {
            self.pipe: {
                "image_name": self.pipe,
                "image_tag": "latest",
                "dockerfile_path": f"pipelines/{self.pipe}/Dockerfile",
                "repository_id": os.environ.get("BUCKET_NAME"),
                "project_id": os.environ.get("PROJECT_ID"),
                "region": os.environ.get("REGION"),
            }
        }
        self.project_config = {
            # Environment Variables
            "project_id": os.environ.get("PROJECT_ID"),
            "project_number": os.environ.get("PROJECT_NUMBER"),
            "region": os.environ.get("REGION"),
            "service_account": os.environ.get("SERVICE_ACCOUNT"),
            "bucket_name": os.environ.get("BUCKET_NAME"),
            # Note: match directory names with pipeline names in pipelines directory
            "directories": self.pipe,
        }
        # Initialzie Cloud Storage Client
        self.storage_client = storage.Client()
        # Initialize Artifact Registry Client
        self.artifactregistry_client = artifactregistry.ArtifactRegistryClient()
        # Initialize Project Config
        self._project_config = ProjectConfig(config=self.project_config)
        # Initialize Artifact Registry Config
        self.artifactregistry_config = ArtifactRegistryConfig(
            client=self.artifactregistry_client, config=self.project_config
        )
        # Initialize Cloud Storage Config
        self.cloud_storage_config = CloudStorageConfig(
            client=self.storage_client, config=self.project_config
        )
        # Initialize Artifact Registry Config
        self.artifactregistry_config = ArtifactRegistryConfig(
            client=self.artifactregistry_client, config=self.project_config
        )

    def enable_resources(self):
        """
        Enable the required resources for the pipeline.
        """

        # Enable APIs
        self._project_config.enable_apis()

    def set_up_storage(self):
        """
        Set up the Cloud Storage bucket and template directories.
        """
        self.cloud_storage_config.create_bucket().template_directories()

    def set_up_artifact_registry(self):
        """
        Set up the Artifact Registry repository.
        """
        self.artifactregistry_config.create_repository()

    def create_container(self):
        """
        Create the Docker container.
        """
        print(f"Creating container with args: {self.container_args[self.pipe]}")
        docker_config = DockerConfig(config=self.container_args[self.pipe])
        docker_config.create_container()

    def define_components(self):
        """
        Define the components of the pipeline.
        """
        # Run definition script for each component in the pipeline
        for component in os.listdir(f"pipelines/{self.pipe}/components"):
            if component.endswith(".py"):
                os.system(
                    f"python pipelines/{self.pipe}/components/{component}/definition.py"
                )

    def define_pipeline(self):
        """
        Define the pipeline by running the definition script.
        """

        os.system(f"python pipelines/{self.pipe}/definition.py")

    def run_pipeline(self):
        """
        Run the pipeline by running the run script.
        """
        os.system(f"python pipelines/{self.pipe}/run.py")

    def magic(self, setup: bool = True, build: bool = True):
        """
        End-to-end pipeline creation and execution process.
        Args:
            setup (bool): Whether to set up the resources before running the pipeline.
        Methods:
            enable_resources: Enable the required resources for the pipeline.
            set_up_storage: Set up the Cloud Storage bucket and template directories.
            set_up_artifact_registry: Set up the Artifact Registry repository.
            create_container: Create the Docker container.
            define_pipeline: Define the pipeline by running the definition script.
            run_pipeline: Run the pipeline by running the run script.
        """
        # Enable the required resources
        if setup:
            self.enable_resources()
            self.set_up_storage()
            self.set_up_artifact_registry()

        # Create the Docker container
        if build:
            self.create_container()

        # Define the pipeline and run it
        self.define_components()
        self.define_pipeline()
        self.run_pipeline()


```

## Custom Containers

To get full control over the prediction process, you can use a custom container. The container is generated using a Dockerfile similar to the following:

```Dockerfile
# Multistage Dockerfile to build the production image
FROM europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest

# ARG PIPELINE_NAME to match the pipeline name in the pipelines directory
ARG PIPELINE_NAME
ARG PROJECT_ID
ARG REGION
ARG BUCKET_NAME

# Set the working directory to the pipeline name
WORKDIR /${PIPELINE_NAME}

# copy the pipeline code to the container
COPY pipelines/${PIPELINE_NAME}/ .

# Set ENV variables based on ARGS
ENV PROJECT_ID=$PROJECT_ID
ENV REGION=$REGION
ENV BUCKET_NAME=$BUCKET_NAME
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=service-account.json
ENV FLASK_APP=./app/app.py

# Install the pipeline code
RUN pip install --upgrade pip
RUN pip install -e .

# Expose port 8080
EXPOSE 8080
ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=8080"]
```

## Serving predictions with Custom Containers

When using custom container:

1. You have to install all the dependencies
2. You have to copy in the container all the src code
3. You have to copy in the container all the artifacts

The term artifact can generally mean:

1. Models in .joblib or pickle format
2. Metadata in yaml or json format

You could also download the model on runtime, but is not recommended as you will slow down the prediction time.

If lineage of the model is mandatory you should create a function to download from cloud storage the model in the container.

## Requirments of Custom Containers

It is suggested to use prebuilt container based on your specific framework and build on top of it.
The src code will host your business logic but a server is required.
Flask is suggested.
The requirements for a serving application are:

1. health route
2. prediction route

Tutorial:
https://www.youtube.com/watch?v=brNMT7Snlh0

## Online Predictions

Online Predictions are made by using an Endpoint.
As custom container are the main choice, the steps are the following:

1. Create an Endpoint ( once )
2. Upload Model in Model Registry
3. Deploy Model to Endpoint
4. Make prediction by post requests

Example of serving custom container:

```bash
├── pipeline name/# the pipeline name is deployment
│   ├── app/
│   │   ├── model/ # directory with the model files
│   │   ├── app.py # Flask app
│   │   ├── predict.py/ # File with the Model Pipeline class to make predictions
```

Note:
The best way to use the model folder is to place the actual model file inside the container since you do not download at runtime any model.
If you prefer a clear linage of models, you can download the model from a MLflow Registry/Cloud Storage and place it inside the container using the SDK.

```python
@dataclass
class ModelLoader:
    pipeline_name: str
    model_name: str
    bucket_name: str
    destination_file_name: str = "./app/model/model.joblib"

    def download_model(self):
        source_blob_name = (
            f"{self.pipeline_name}/artifacts/model/{self.model_name}.joblib"
        )
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(self.destination_file_name)
            logging.info("Model downloaded from Cloud Storage.")
            load(self.destination_file_name)
            logging.info("Model loaded from Cloud Storage.")
        except Exception as e:
            logging.error("Error downloading model from Cloud Storage: %s", e)
        # Wait for the model to load
        time.sleep(0.1)
```

Example of online prediction using flask `app.py`

```python
import os
from flask import Flask, jsonify, request, json
import pandas as pd
from predict import ModelPipeline

app = Flask(__name__)
AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


@app.route("/health")
def health():
    """Health endpoint.


    Returns:
        response: health response
    """
    return "OK", 200


@app.route("/predict", methods=["POST", "GET"])
def predict():
    """Predict endpoint.


    Args:
        request (post): post request with instances in body


    Returns:
        response: prediction response
    """

    predictor = ModelPipeline()
    instances = request.get_json()["instances"]
    results = predictor.predict(data=instances)

    # Format Vertex AI prediction response
    predictions = [
        {"probability_negative": result[0], "probability_positive": result[1]}
        for result in results
    ]

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run()
```
