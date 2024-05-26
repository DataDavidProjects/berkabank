from typing import List
import joblib
from dataclasses import dataclass


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

    def load_model(self):
        """Load model from disk"""
        model = joblib.load(self.model_path)

        return model

    def __post_init__(self):
        self.model = self.load_model()

    def processing(self, data):
        """Preprocess data"""
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
