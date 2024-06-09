from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import re
from dataclasses import dataclass
from typing import Dict, Union, List
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np


def parse_aggregation(aggregation):
    """Parse the name of an aggregation to get the feature, category, function, and time period.

    Args:
        aggregation (str): Name of the aggregation.

    Returns:
        tuple: feature, category, function, time period.
    """
    match = re.match(r"f_(\w+)__rolling_(\w+)_(\d+)_days", aggregation)
    feature, func, time_period = match.groups()
    time_period = int(time_period)

    return feature, func, time_period


def parse_aggregation_extended(aggregations) -> List[str]:
    extended_aggregations = []
    for aggregation in aggregations:
        if "ratio" in aggregation:
            extended_aggregations += aggregation.split("_ratio_")
            aggregations.remove(aggregation)
        else:
            extended_aggregations.append(aggregation)
    return extended_aggregations


def report_drivers(aggregations) -> pd.DataFrame:
    extended_agg = parse_aggregation_extended(aggregations)
    parsed_aggregations = [parse_aggregation(agg) for agg in extended_agg]
    return pd.DataFrame(
        parsed_aggregations,
        columns=["Base Feature", "Aggregation Function", "Time Period - Days"],
    )


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
        self.y_pred = self.y_proba[:, 1] > self.cutoff_score

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
            "ROC AUC Score": roc_auc_score(self.y, self.y_pred),
            "Accuracy": accuracy_score(self.y, self.y_pred),
            "Precision": precision_score(self.y, self.y_pred),
            "Recall": recall_score(self.y, self.y_pred),
            "F1 Score": f1_score(self.y, self.y_pred),
            "F0.5 Score": fbeta_score(self.y, self.y_pred, beta=0.5),
            "F2 Score": fbeta_score(self.y, self.y_pred, beta=2),
        }

        metrics_df = pd.DataFrame(metrics_dict, index=[0])

        return metrics_df.rename_axis("Model", axis=0)

    def run(self) -> None:
        """Run the model evaluation process.

        Returns:
            None
        """
        cm = self.confusion_matrix()
        print("Confusion Matrix:\n", cm)
        return self.report()
