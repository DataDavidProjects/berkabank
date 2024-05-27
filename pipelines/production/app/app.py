import os
from flask import Flask, jsonify, request, json
from google.auth import default, exceptions
import pandas as pd
from predict import ModelPipeline, ModelLoader, ValidationCheck
from google.cloud import storage
import joblib
from dotenv import load_dotenv, find_dotenv
import time
from loguru import logger


# Start Flask Server
app = Flask(__name__)
AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


# Load ENV
load_dotenv(find_dotenv())
print("ENV loaded.")
# Define the details component
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_NAME = "berkamodel"
PIPELINE_NAME = "production"
INDEX = "account_id"

# Download model from cloud storage
DESTINATION_FILE_NAME = "./app/model/model.joblib"
SOURCE_BLOB_NAME = f"{PIPELINE_NAME}/artifacts/model/{MODEL_NAME}.joblib"
# Initialize ModelLoader with the necessary parameters
model_loader = ModelLoader(
    pipeline_name=PIPELINE_NAME, model_name=MODEL_NAME, bucket_name=BUCKET_NAME
)

# Download model
model_loader.download_model()

# Validate model


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
    logger.info("Request received.")
    predictor = ModelPipeline(model_path=DESTINATION_FILE_NAME, index=INDEX)
    logger.info("Model loaded.")
    instances = request.get_json()["instances"]
    logger.info("Instances received.")

    # Validate instances
    validator = ValidationCheck(instances=instances, model_path=DESTINATION_FILE_NAME)
    instances = validator.validate_instances()
    logger.info("Instances validated.")

    # Handle input cases
    try:
        features_names = predictor.model.feature_names_in_.tolist()
        data = pd.DataFrame(instances)[features_names]
    except KeyError:
        data = instances
    logger.info("Data prepared.")

    results = predictor.predict(data=data)
    logger.info("Prediction done.")

    # Format Vertex AI prediction response
    predictions = [
        {"probability_negative": result[0], "probability_positive": result[1]}
        for result in results
    ]

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)
