from typing import List
import joblib
from dataclasses import dataclass
import pandas as pd
from typing import Union
from loguru import logger
from google.cloud import storage
from joblib import load
import time
import logging


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


@dataclass
class ValidationCheck:
    instances: any
    model_path: str = "./model/model.joblib"

    def __post_init__(self):
        logger.info("Instances shape: \n", self.instances)
        self.model = self.load_model()

    def load_model(self):
        """Load model from disk"""
        model = joblib.load(self.model_path)
        return model

    @property
    def model_features(self):
        return list(self.model.feature_names_in_)

    def validate_instances(self):
        if len(self.instances) > 0:
            idx = 1
        else:
            idx = 0
        # Index is offseting the entries of the instances
        if len(self.model_features) + 1 == len(self.instances[idx]) and isinstance(
            self.instances, List
        ):
            logger.info("Removing possible index from instances")
            self.instances = [instance[1:] for instance in self.instances]
            logger.info("New Instances: \n", self.instances)

        return self.instances


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
