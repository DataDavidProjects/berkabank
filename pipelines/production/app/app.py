import os
from flask import Flask, jsonify, request, json
from google.auth import default, exceptions
import pandas as pd
from predict import ModelPipeline
from google.cloud import storage
import joblib
from dotenv import load_dotenv, find_dotenv


# Load ENV
load_dotenv(find_dotenv())

# Define the details component
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_NAME = "berkamodel"
PIPELINE_NAME = "production"

# Download model from cloud storage
source_blob_name = f"{PIPELINE_NAME}/artifacts/model/{MODEL_NAME}.joblib"
destination_file_name = "app/model/model.joblib"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)

print("Model downloaded from Cloud Storage.")

# Start Flask Server
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
