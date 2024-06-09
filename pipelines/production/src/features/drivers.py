from dataclasses import dataclass
import pandas as pd
import numpy as np
import re
from typing import List, Union, Dict, Any
from src.features.engineering import make_aggregations


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


def make_drivers(
    df: pd.DataFrame,
    drivers: List[str],
    index: str = "account_id",
) -> pd.DataFrame:
    """Create rolling aggregations for the given dataframe based on the name of the aggregation.

    Args:
        df (pd.DataFrame): DataFrame containing eos balance data.
        drivers (List[str]): Feature names to create rolling aggregations for.

    Returns:
        pd.DataFrame: DataFrame with rolling aggregations for each period.
    """
    features = []
    ratios = []
    for aggregation in drivers:
        decomposition = decompose_aggregation(aggregation)

        if "ratio" in decomposition:
            numerator = decomposition["ratio"]["numerator"]["feature"]
            denominator = decomposition["ratio"]["denominator"]["feature"]
            features.append(numerator)
            features.append(denominator)
            ratios.append(decomposition)
        else:
            features.append(decomposition["feature"])

    print(features)
    feature_matrix = (
        make_aggregations(df, features)
        .groupby(index)[features]
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
        f"f_{major}_{minor}" for major, minor in feature_matrix.columns.values
    ]

    numerators = [
        n
        for n in feature_matrix.columns
        if any(
            numerator in n
            for ratio in ratios
            for numerator in [ratio["ratio"]["numerator"]["feature"]]
        )
    ]
    denominators = [
        d
        for d in feature_matrix.columns
        if any(
            denominator in d
            for ratio in ratios
            for denominator in [ratio["ratio"]["denominator"]["feature"]]
        )
    ]
    EPS = np.finfo(float).eps
    for f_numerator in numerators:
        for f_denominator in denominators:
            feature_matrix[f"{f_numerator}_ratio_{f_denominator}"] = feature_matrix[
                f_numerator
            ] / (feature_matrix[f_denominator] + EPS)

    return feature_matrix.loc[:, drivers]


@dataclass
class EODBDrivers:
    eod_balance_preprocessed: pd.DataFrame
    drivers: List[str]

    def run(self):
        print("----- Running EODBDrivers...")

        return make_drivers(self.eod_balance_preprocessed, self.drivers)
