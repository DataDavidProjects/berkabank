# Fundamentals of Lazy Pipelines

A LazyPipeline is a pipeline that follows specific conventions and allows users to move their source code to a VertexAI pipeline in less time as possible and with full control of business logic.

1. Naming conventions for files and directories
2. Components
3. Pipelines

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
