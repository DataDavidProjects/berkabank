# Core Concepts

## Custom Containers

To get full control over the prediction process, you can use a custom container.

The container is generated using a Dockerfile similar to the following

```dockerfile
FROM europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest # Prebuilt container


# Set the working directory on the container
WORKDIR /

# Create the pipeline directory (named deployment) on the container
RUN mkdir /deployment
# Copy all the files from the pipeline directory to the container directory
COPY pipelines/deployment/ /deployment/ # including metadata,artifacts and scripts

# Install the dependencies and src code
RUN pip install --upgrade pip
RUN cd /deployment/ && pip install -e .

# Set environment variables for flask endpoint
ENV FLASK_APP=/deployment/app/app.py
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
    features_names = predictor.model.feature_names_in_.tolist()
    instances = request.get_json()["instances"]
    data = pd.DataFrame(instances)[features_names]
    results = predictor.predict(data=data)

    # Format Vertex AI prediction response
    predictions = [
        {"probability_negative": result[0], "probability_positive": result[1]}
        for result in results
    ]

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
    )


if __name__ == "__main__":
    app.run()

```
