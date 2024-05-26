# Standard library imports
import os
import json
import datetime
from collections import defaultdict
from typing import Union
from dataclasses import dataclass

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage
from flask import Flask, jsonify, request
from dotenv import load_dotenv, find_dotenv
import joblib


input_bucket_path = "gs://berkabank/production/data/"

bucket_name = "berkabank"
source_blob_name = "production/artifacts/model/berkamodel.joblib"
destination_file_name = "./berkamodel.joblib"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
joblib.load(destination_file_name)


@dataclass
class ModelPipeline:
    """Pipeline for prediction
    Args:
        model_path (str): Path to the model file

    Attributes:
        model_path (str): Path to the model file

    Methods:
        load_model(): Load the model from disk
        processing(data): Preprocess the data
        inference(data): Predict using the model
        postprocessing(prediction): Postprocess the prediction
        predict(data): Predict using the model

    Returns:
        output: Prediction output
    """

    model_path: str = "./model/model.joblib"
    index: str = None

    def load_model(self):
        """Load model from disk"""
        model = joblib.load(self.model_path)

        return model

    def __post_init__(self):
        self.model = self.load_model()

    def processing(self, data, index=index):
        """Preprocess data"""
        if index and isinstance(data, Union[pd.DataFrame, pd.Series]):
            data = data.set_index(index)
        return data

    def inference(self, data):
        """Predict using the model"""
        prediction = self.model.predict_proba(data)
        return prediction

    def postprocessing(self, prediction):
        """Postprocess prediction"""
        return prediction

    def predict(self, data):
        """Predict using the model"""

        data = self.processing(data)
        prediction = self.inference(data)
        output = self.postprocessing(prediction)
        return output


# Load ENV
load_dotenv(find_dotenv())

# Define the details component
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_NAME = "berkamodel"

# Download model from cloud storage
# source_blob_name = f"production/artifacts/model/{MODEL_NAME}.joblib"
# destination_file_name = "./model.joblib"
# storage_client = storage.Client()
# bucket = storage_client.bucket(BUCKET_NAME)
# blob = bucket.blob(source_blob_name)
# blob.download_to_filename(destination_file_name)
# joblib.load(destination_file_name)

# Start Flask Server
app = Flask(__name__)
# AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
# AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


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

    predictor = ModelPipeline("./berkamodel.joblib", index="account_id")

    features_names = predictor.model.feature_names_in_.tolist()
    instances = request.get_json()["instances"]
    data = pd.DataFrame(instances)[features_names]
    results = predictor.predict(data=data)  # tobe score method

    # Format Vertex AI prediction response
    predictions = [
        {"probability_negative": result[0], "probability_positive": result[1]}
        for result in results
    ]

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)


if __name__ == "__main__":
    app.run()
