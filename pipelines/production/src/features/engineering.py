from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any, Callable
import pandas as pd
import re
import warnings
import random
import numpy as np
from typing import Callable


# Suppress the PerformanceWarning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def get_function_name(func):
    if callable(func):
        return func.__name__
    else:
        return func


def create_aggregations(
    max_t: int = 60,
    min_t: int = 5,
    step: int = 5,
    agg_funcs: List[str] = ["mean", "max", "sum", "std", "median"],
    features: List[str] = ["daily_amount_flow", "n_transactions"],
) -> List[str]:
    """Create aggregation names based on the given parameters.

    Args:
        max_t (int, optional): Maximum time period. Defaults to 60.
        min_t (int, optional): Minimum time period. Defaults to 5.
        agg_funcs (List[str], optional): Aggregation functions. Defaults to ["mean", "max", "sum", "std", "median"].
        category (List[str], optional): Categories. Defaults to ["inflow", "outflow"].
        features (List[str], optional): Features. Defaults to ["daily_amount_flow"].

    Returns:
        List[str]: Names of the aggregations.
    """
    return [
        f"f_{feature}__rolling_{func}_{time_period}_days"
        for feature in features
        for time_period in range(min_t, max_t + 1, step)
        for func in agg_funcs
    ]


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


def make_aggregations(
    df: pd.DataFrame,
    agg_funcs: List[str],
    aggregation_mapping: Dict[str, str] = None,
    index: str = "account_id",
) -> pd.DataFrame:
    """Create rolling aggregations for the given dataframe based on the name of the aggregation.

    Args:
        df (pd.DataFrame): DataFrame containing eos balance data.
        agg_funcs (List[str]): Feature names to create rolling aggregations for.

    Returns:
        pd.DataFrame: DataFrame with rolling aggregations for each period.
    """
    if aggregation_mapping is None:
        aggregation_mapping = {
            "mean": "mean",
            "min": "min",
            "max": "max",
            "sum": "sum",
            "std": "std",
            "median": "median",
            "skew": "skew",
            "kurtosis": "kurt",
            **{f"percentile_{p}": "quantile" for p in range(5, 100)},
        }

    for aggregation in agg_funcs:
        # print(f"Applying aggregation function: {aggregation}")
        feature, func, time_period = parse_aggregation(aggregation)
        agg = aggregation_mapping.get(func)
        if agg is None:
            raise ValueError(f"Function {func} not found in the function mapping.")

        params = {}
        if func.startswith("percentile"):

            percentile = int(func.split("_")[1]) / 100
            params[agg] = percentile
        # Suppress the FutureWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df[aggregation] = getattr(
                df.groupby(index)[feature].rolling(time_period), agg
            )(**params).reset_index(level=0, drop=True)

    return df


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
    aggregations: List[str]
    column_mapping: dict
    index: str = "account_id"

    def run(self):
        print("----- Running EODBFeatures...")
        feature_matrix = (
            make_aggregations(self.eod_balance_training, self.aggregations)
            .groupby(self.column_mapping.get(self.index))[self.aggregations]
            .agg(
                [
                    "sum",
                    "size",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "skew",
                ]
            )
        )

        feature_matrix.columns = [
            f"f_{measure}_{fun}" for measure, fun in feature_matrix.columns.values
        ]
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

    def run(self):
        print("--- Running DerivedFeatures...")
        # Small constant to avoid division by zero
        eps = np.finfo(float).eps
        # Rank columns by kurtosis

        # Get all possible pairs of columns
        pairs = [
            (col1, col2)
            for col1 in self.df.columns
            for col2 in self.df.columns
            if col1 != col2
        ]

        if self.n:
            pairs = self.strategies[self.strategy](pairs)

            # Select the first n pairs
            pairs = pairs[: self.n]

        # Iterate over the selected pairs
        for num, den in pairs:
            # Create new column with division result
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
