from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any, Callable
import pandas as pd
import re
import warnings

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
    column_mapping: dict
    aggregations: List[str]

    def run(self):
        print("----- Running EODBFeatures...")
        feature_matrix = (
            make_aggregations(self.eod_balance_training, self.aggregations)
            .groupby(self.column_mapping.get("account_id"))[self.aggregations]
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
            f"f_{col[0]}_{col[1]}" for col in feature_matrix.columns.values
        ]
        eodb_features_output = feature_matrix.copy()
        print("----- EODBFeatures completed.")
        return eodb_features_output


@dataclass
class PrimaryFeatures:
    incident_features: IncidentFeatures
    eodb_features: EODBFeatures

    def __post_init__(self):
        print("--- Initializing PrimaryFeatures...")
        self.incident_features_output = self.incident_features.run()
        self.eodb_features_output = self.eodb_features.run()
        print("--- PrimaryFeatures initialized.")

    def run(self):
        print("--- Running PrimaryFeatures...")
        primary_features_output = pd.DataFrame()
        print("--- PrimaryFeatures completed.")
        return primary_features_output


@dataclass
class DerivedFeatures:
    primary_features: PrimaryFeatures

    def __post_init__(self):
        print("--- Initializing DerivedFeatures...")
        self.primary_features_output = self.primary_features.run()
        print("--- DerivedFeatures initialized.")

    def run(self):
        print("--- Running DerivedFeatures...")
        derived_features = pd.DataFrame()
        print("--- DerivedFeatures completed.")
        return derived_features


@dataclass
class FeatureEngineering:
    primary_features: PrimaryFeatures
    derived_features: DerivedFeatures

    def run(self):
        print("Running FeatureEngineering...")
        self.derived_features.run()
        print("FeatureEngineering completed.")
