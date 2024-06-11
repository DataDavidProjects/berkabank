from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any, Callable
import pandas as pd
import re
import warnings
import random
import numpy as np
from src.features.utils import *

# Suppress the PerformanceWarning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass
class IncidentFeatures:
    incidents: pd.DataFrame
    column_mapping: dict

    def run(self):
        print("----- Running IncidentFeatures...")
        incident_features = pd.DataFrame()
        print("----- IncidentFeatures completed.")
        return incident_features


@dataclass
class EODBFeatures:
    eod_balance_training: pd.DataFrame
    column_mapping: dict
    index: str = "account_id"

    def run(self):
        print("----- Running EODBFeatures...")
        feature_matrix = compute_aggregations(
            eod_balance=self.eod_balance_training,
        )
        eodb_features_output = feature_matrix.copy()
        print("----- EODBFeatures completed.")
        return eodb_features_output


@dataclass
class RatioFeatures:
    """A class to generate ratio features from a DataFrame.

    To avoid not necessary computation, the ratios are generated using a kurtosis rank.
    The assumption is that flat curves are a blend of 2 possible distrubitions.
    The lowest in the rank are selected and different ratio pairs are created

    Attributes:
        df: A pandas DataFrame from which to generate ratio features.
        n: An optional integer specifying the number of pairs to select. If not specified, all pairs are used.

    Returns:
        A new DataFrame with the original data and the new ratio features.
    """

    df: pd.DataFrame
    n: int = None
    strategy: str = "random"

    def random_strategy(self, pairs):
        return random.sample(pairs, self.n)

    def kurtosis_strategy(self, pairs):
        ranked_columns = self.df.kurt().sort_values(ascending=True).index.tolist()
        return [
            (col1, col2)
            for col1 in ranked_columns
            for col2 in ranked_columns
            if col1 != col2
        ]

    def __post_init__(self):
        self.strategies = {
            "random": self.random_strategy,
            "kurtosis": self.kurtosis_strategy,
        }

    def create_pairs(self):
        # Get all possible pairs of columns
        pairs = [
            (col1, col2)
            for col1 in self.df.columns
            for col2 in self.df.columns
            if col1 != col2
        ]
        return pairs

    def run(self):
        print("--- Running DerivedFeatures...")

        pairs = self.create_pairs()
        # Create ratio based on the selected strategy
        if self.n:
            # Select the first n pairs
            pairs = self.strategies[self.strategy](pairs)
            pairs = pairs[: self.n]

        # Create new column with division result
        eps = np.finfo(float).eps
        for num, den in pairs:
            self.df[f"{num}_ratio_{den}"] = self.df[num] / (self.df[den] + eps)

        ratiofeatures = self.df.copy()
        print("--- DerivedFeatures completed.")
        return ratiofeatures


@dataclass
class PrimaryFeatures:
    df: pd.DataFrame
    steps: Dict[str, Callable]

    def run(self):
        print("--- Running PrimaryFeatures...")
        for step in self.steps:
            self.df = self.steps[step].run(self.df)

        primary_features_output = self.df.copy()
        print("--- PrimaryFeatures completed.")
        return primary_features_output


@dataclass
class DerivedFeatures:
    """Compute features based on the existing features.

    Returns:
        pd.Dataframe: Dataframe with derived features.
    """

    df: pd.DataFrame
    steps: Dict[str, Callable]

    def run(self):
        print("--- Running DerivedFeatures...")
        for step in self.steps:
            self.df = self.steps[step].run(self.df)

        derived_features_output = self.df.copy()
        print("--- DerivedFeatures completed.")
        return derived_features_output


@dataclass
class FeatureImputer:
    """Impute missing values in the dataframe.

    Returns:
        pd.Dataframe: Imputed dataframe.
    """

    df: pd.DataFrame
    filler: float = -1

    def run(self):
        print("--- Running FeatureImputer...")
        imputed_df = self.df.fillna(self.filler)
        print("--- FeatureImputer completed.")
        return imputed_df
