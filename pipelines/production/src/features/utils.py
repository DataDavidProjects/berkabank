from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any, Callable
import pandas as pd
import re
import warnings
import random
import numpy as np


def get_function_name(func):
    if callable(func):
        return func.__name__
    else:
        return func


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


def extract_ratio_dependency(aggregation) -> List[str]:
    if "_ratio_" in aggregation:
        parts = aggregation.split("_ratio_")
        matches = [re.search("f_(.*)_", part).group(1) for part in parts]
        return matches
    else:
        pass


def extract_function_for_aggregation(aggregation) -> str:
    return re.split("_", aggregation)[-1]


def extract_aggregation_dependency(aggregation, n=1):
    match = re.search("f_(.*)_", aggregation)
    if match:
        return match.group(n)


def decompose_aggregation(aggregation):
    if "_ratio_" in aggregation:
        parts = aggregation.split("_ratio_")
        return {
            "ratio": {
                "numerator": decompose_aggregation(parts[0]),
                "denominator": decompose_aggregation(parts[1]),
            }
        }
    else:
        return {
            "feature": extract_aggregation_dependency(aggregation),
        }


def get_dependencies(aggregations):
    dependencies = []
    for aggregation in aggregations:
        decomposed = decompose_aggregation(aggregation)
        if "ratio" in decomposed:
            dependencies.append(decomposed["ratio"]["numerator"]["feature"])
            dependencies.append(decomposed["ratio"]["denominator"]["feature"])
        else:
            dependencies.append(decomposed["feature"])
    return dependencies


def extract_aggregations_from_string(s):
    # Define all possible aggregations
    all_aggregations = ["sum", "size", "mean", "std", "min", "max", "skew"]

    # Extract the necessary aggregations
    necessary_aggregations = [agg for agg in all_aggregations if agg in s]

    return necessary_aggregations


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


def decompose_feature_string(feature_string: str) -> Dict[str, Any]:
    """Decompose a feature string into its components.

    Args:
        feature_string (str): Feature string to decompose.

    Returns:
        Dict[str, Any]: Dictionary containing the components of the feature string.
    """
    # Initialize the result dictionary
    result = {}

    # Check if the feature string contains a ratio
    if "_ratio_" in feature_string:
        # Split the feature string into numerator and denominator
        numerator, denominator = feature_string.split("_ratio_")

        # Decompose the numerator and denominator
        result["numerator"] = decompose_feature_string(numerator)
        result["denominator"] = decompose_feature_string(denominator)
    else:
        # Extract the feature, function, and time period using regular expressions
        match = re.match(r"f_(.*?)__(rolling_(.*?)_(\d+))_days_(.*)", feature_string)
        if match:
            feature = match.group(1)[2:]
            rolling_statistic = match.group(3)
            time_period = int(match.group(4))
            aggregation = match.group(5)

            result["feature"] = feature
            result["rolling_statistic"] = rolling_statistic
            result["time_period"] = time_period
            result["aggregation"] = aggregation

    return result


def create_rolling_features(
    max_t: int = 60,
    min_t: int = 5,
    step: int = 5,
    rolling_statistics: List[str] = None,
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
    if rolling_statistics is None:
        rolling_statistics = [
            "mean",
            "min",
            "max",
            "sum",
            "std",
            "median",
            "skew",
        ]
    return [
        f"f_{feature}__rolling_{func}_{time_period}_days"
        for feature in features
        for time_period in range(min_t, max_t + 1, step)
        for func in rolling_statistics
    ]


def create_aggregations(
    rolling_features: List[str], aggregation_functions: List[str] = None
) -> List[str]:
    if aggregation_functions is None:
        aggregation_functions = [
            "mean",
            "min",
            "max",
            "sum",
            "std",
            "median",
            "skew",
            "kurt",
        ]
    return [
        f"f_{feature}_{aggregation_function}"
        for feature in rolling_features
        for aggregation_function in aggregation_functions
    ]


def decompose_aggregation_name(feature_name: str):
    # Define the regular expression pattern
    pattern = r"f_f_(.+)__rolling_(.+)_(\d+)_days_(.+)"

    # Use re.match to match the pattern in the feature name
    match = re.match(pattern, feature_name)

    # Extract the components from the match object
    feature_column = match.group(1)
    rolling_statistic = match.group(2)
    window = int(match.group(3))
    aggregation = match.group(4)

    return {
        "feature_column": feature_column,
        "rolling_statistic": rolling_statistic,
        "window": window,
        "aggregation": aggregation,
    }


def compute_aggregation(
    eod_balance: pd.DataFrame,
    aggregation_column: str,
    function_mapping: Dict[str, str] = None,
    date_column: str = "balance_date",
    index_column: str = "account_id",
) -> pd.DataFrame:

    # Define the mapping of aggregation functions to callables
    if function_mapping is None:
        function_mapping = {
            "mean": "mean",
            "min": "min",
            "max": "max",
            "sum": "sum",
            "std": "std",
            "median": "median",
            "skew": "skew",
            "kurt": "kurt",
            **{f"percentile_{p}": "quantile" for p in range(5, 100)},
        }

    # Decompose the aggregation column name
    decomposition = decompose_aggregation_name(aggregation_column)
    feature_column = decomposition["feature_column"]
    rolling_statistic = function_mapping.get(decomposition["rolling_statistic"])
    window = decomposition["window"]
    aggregation = function_mapping.get(decomposition["aggregation"])

    # Initialize parameters for rolling and aggregation functions
    rolling_params = {}
    aggregation_params = {}
    # Handle percentile case for rolling statistic
    if decomposition["rolling_statistic"].startswith("percentile"):
        match = re.search(r"percentile_(\d+)", decomposition["rolling_statistic"])
        if match is None:
            raise ValueError(
                f"Invalid format for rolling_statistic: {decomposition['rolling_statistic']}"
            )
        percentile = int(match.group()) / 100
        rolling_params["q"] = percentile

    # Handle percentile case for aggregation
    if decomposition["aggregation"].startswith("percentile"):
        match = re.search(r"percentile_(\d+)", decomposition["aggregation"])
        if match is None:
            raise ValueError(
                f"Invalid format for aggregation: {decomposition['aggregation']}"
            )
        percentile = int(match.group()) / 100
        aggregation_params["q"] = percentile

    # Order eod balance by date
    ranked_period = eod_balance.sort_values(date_column, ascending=True)
    # Group by account_id and select a feature column
    feature_grouped = ranked_period.groupby(index_column)[feature_column]

    # Handle Kurtosis case for rolling and aggregation functions
    if decomposition["rolling_statistic"] == "kurt":
        # Compute rolling statistic of period window for the feature column, for each account_id. inject params if needed
        feature_grouped_rolling_statistic = feature_grouped.rolling(window).apply(
            pd.Series.kurt
        )
    else:
        feature_grouped_rolling_statistic = getattr(
            feature_grouped.rolling(window), rolling_statistic
        )(**rolling_params)

    if decomposition["aggregation"] == "kurt":
        # Aggregate the rolling statistic by account_id using an aggregation function. inject params if needed
        aggregation_rolling_statistic = feature_grouped_rolling_statistic.groupby(
            index_column
        ).apply(pd.Series.kurt)
    else:
        # Aggregate the rolling statistic by account_id using an aggregation function. inject params if needed
        aggregation_rolling_statistic = getattr(
            feature_grouped_rolling_statistic.groupby(index_column), aggregation
        )(**aggregation_params)

    # Assign the column names for each aggregation_rolling_statistic
    aggregation_rolling_statistic.name = (
        f"f_f_{feature_column}__rolling_{rolling_statistic}_{window}_days_{aggregation}"
    )

    return aggregation_rolling_statistic


def compute_aggregations(
    eod_balance: pd.DataFrame,
    min_t: int = 10,
    max_t: int = 90,
    step: int = 10,
    rolling_statistics: List[str] = [
        "mean",
        "min",
        "max",
        "sum",
        "std",
        "median",
        "skew",
        "kurt",
    ],
    aggregations: List[str] = ["mean", "min", "max", "sum", "std", "median", "skew"],
    feature_columns: List[str] = ["daily_amount_inflow", "daily_amount_outflow"],
    index_column: str = "account_id",
):

    df_agg = pd.concat(
        [
            eod_balance.groupby(index_column)
            .rolling(n)
            .agg(
                {
                    feature_column: rolling_statistics
                    for feature_column in feature_columns
                }
            )
            .groupby(index_column)
            .agg(aggregations)
            for n in range(min_t, max_t, step)
        ],
        axis=1,
    )

    columns = [
        f"f_f_{feature_column}__rolling_{rs}_{window}_days_{agg}"
        for feature_column in feature_columns
        for window in range(min_t, max_t, step)
        for rs in rolling_statistics
        for agg in aggregations
    ]
    df_agg.columns = columns

    return df_agg


def compute_driver_basic(
    eod_balance: pd.DataFrame, driver: str, index_column: str = "account_id"
) -> pd.Series:

    decomposition = decompose_aggregation_name(
        "f_f_daily_amount_inflow__rolling_mean_10_days_mean"
    )
    feature_column = decomposition["feature_column"]
    rolling_statistic = decomposition["rolling_statistic"]
    window = decomposition["window"]
    aggregation = decomposition["aggregation"]

    series = (
        eod_balance.groupby(index_column)
        .rolling(window)
        .agg({feature_column: rolling_statistic})
        .groupby(index_column)
        .agg(aggregation)
    )

    series.columns = [driver]
    return series
