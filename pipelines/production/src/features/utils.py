import pandas as pd
from typing import Dict, List, Any, Callable
import pandas as pd
import re
import numpy as np


def get_function_name(func):
    if callable(func):
        return func.__name__
    else:
        return func


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


def parse_aggregation(aggregation):
    match = re.match(r"f_(\w+)__rolling_(\w+)_(\d+)_days", aggregation)
    if match:
        feature, func, time_period = match.groups()
        time_period = int(time_period)
        return feature, func, time_period
    else:
        return None


def report_drivers(aggregations) -> pd.DataFrame:
    ratio_drivers = [col for col in aggregations if "ratio" in col]
    basic_drivers = [col for col in aggregations if "ratio" not in col]

    parsed_basic_aggregations = [parse_aggregation(agg) for agg in basic_drivers]

    ratio_parts = []
    for ratio in ratio_drivers:
        num, den = ratio.split("_ratio_")
        ratio_parts.append(parse_aggregation(num))
        ratio_parts.append(parse_aggregation(den))

    parsed_aggregations = parsed_basic_aggregations + ratio_parts

    return pd.DataFrame(
        parsed_aggregations,
        columns=["Base Feature", "Aggregation Function", "Time Period - Days"],
    )


def compute_driver_basic(
    eod_balance: pd.DataFrame, driver: str, index_column: str = "account_id"
) -> pd.Series:

    decomposition = decompose_aggregation_name(driver)
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
