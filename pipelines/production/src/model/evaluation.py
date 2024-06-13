from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import re
from dataclasses import dataclass
from typing import Dict, Union, List, Tuple
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np


@dataclass
class ModelEvaluationBinaryClassification:
    """Model evaluation class for binary classification.

    Attributes:
        model (BaseEstimator): model to evaluate
        X (pd.DataFrame): input features
        y (np.array): target variable
        test_size (float): proportion of the dataset to include in the test split
        random_state (int): controls the shuffling applied to the data before applying the split

    Methods:
        run(): fit the model and print the evaluation metrics

    Returns:
        Object
    """

    model: BaseEstimator
    y_proba: pd.DataFrame
    y: np.array
    cutoff_score: float = 0.5

    def __post_init__(self):
        self.y_proba_positive = self.y_proba[:, 1]
        self.y_pred = self.y_proba_positive > self.cutoff_score

    def confusion_matrix(self) -> pd.DataFrame:
        """Generate the confusion matrix.

        Returns:
            pd.DataFrame: confusion matrix
        """

        cm = confusion_matrix(self.y, self.y_pred)
        return cm

    def report(self):
        """Report the evaluation metrics.

        Returns:
            pd.Dataframe: metrics dataframe
        """
        metrics_dict = {
            "ROC AUC Score": roc_auc_score(self.y, self.y_proba_positive),
            "Precision": precision_score(self.y, self.y_pred),
            "Recall": recall_score(self.y, self.y_pred),
            "F1 Score": f1_score(self.y, self.y_pred),
            "F0.5 Score": fbeta_score(self.y, self.y_pred, beta=0.5),
            "F2 Score": fbeta_score(self.y, self.y_pred, beta=2),
        }
        try:
            index_metric_df = [self.model.__class__.__name__]
        except:
            index_metric_df = ["Model"]

        metrics_df = pd.DataFrame(metrics_dict, index=index_metric_df)

        return metrics_df.rename_axis("Model", axis=0), classification_report(
            self.y, self.y_pred
        )

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the model evaluation process.

        Returns:
            None
        """
        cm = self.confusion_matrix()
        metrics_df, classification_report = self.report()

        return (cm, metrics_df, classification_report)
