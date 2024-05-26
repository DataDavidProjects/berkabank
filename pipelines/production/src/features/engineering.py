from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any, Callable


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
    aggregations: dict

    def run(self):
        print("----- Running EODBFeatures...")
        eodb_features_output = pd.DataFrame()
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
