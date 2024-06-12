import os
from typing import Dict, List, Any
from kfp import dsl
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kfp import compiler


load_dotenv(find_dotenv())


# Define the details component
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
REPOSITORY = os.environ.get("BUCKET_NAME")  # Match the Bucket name on Artifact Registry

PIPELINE_NAME = (
    Path(__file__).resolve().parents[2].name
)  # Match the directory name of pipeline
COMPONENT_NAME = os.path.basename(os.path.dirname(__file__))  # Match the directory name
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{PIPELINE_NAME}:latest"


@dsl.component(
    base_image=BASE_IMAGE,
)
def feature_engineering_component(
    input_bucket_path: str,
    output_bucket_path: str,
    input_files: Dict[str, str],
    output_files: Dict[str, str],
    column_mapping: Dict[str, Any],
    aggregations: dict,
    eod_balance_training: dsl.Input[dsl.Dataset],
    incidents: dsl.Input[dsl.Dataset],
    training_features_output: dsl.Output[dsl.Dataset],
):
    from src.features.engineering import (
        EODBFeatures,
        RatioFeatures,
    )
    from src.features.selection import (
        FeatureEliminationCoV,
        FeatureEliminationKurtosis,
        FeatureEliminationMissingRate,
        FeatureEliminationPearsonCorr,
        FeatureEliminationPipeline,
        FeatureEliminationVIF,
    )
    from src.features.imputer import FeatureImputer

    import pandas as pd

    # Load the data
    data = {
        "eod_balance_training": pd.read_csv(
            f"{input_bucket_path}04_processing/{input_files['eod_balance_training']}.csv"
        ),
        "accounts": pd.read_csv(
            f"{input_bucket_path}03_primary/{input_files['accounts']}.csv"
        ),
        "incidents": pd.read_csv(
            f"{input_bucket_path}03_primary/{input_files['incidents']}.csv"
        ),
    }

    # EOD Balance Feature Engineering
    eod_balance_training = data["eod_balance_training"].copy()

    eod_balance_training["flow_category"] = (
        eod_balance_training["daily_amount_flow"]
        .gt(0)
        .replace({True: "inflow", False: "outflow"})
    )

    eod_balance_training["daily_amount_inflow"] = eod_balance_training[
        "daily_amount_flow"
    ].clip(lower=0)

    eod_balance_training["daily_amount_outflow"] = (
        eod_balance_training["daily_amount_flow"].clip(upper=0).abs()
    )

    index = ["account_id"]
    FEATURE_BASED = [
        "daily_amount_inflow",
        "daily_amount_outflow",
        "end_of_day_balance",
    ]

    features_selected = eod_balance_training.loc[:, index + FEATURE_BASED]
    eod_feature_matrix = EODBFeatures(
        eod_balance_training=features_selected,
        column_mapping=column_mapping["eod_balance_training"],
        feature_columns=FEATURE_BASED,
    ).run()

    eod_feature_matrix = RatioFeatures(
        df=eod_feature_matrix, n=1000, strategy="kurtosis"
    ).run()

    # Statistical Feature Selection
    feature_elimination_pipeline = FeatureEliminationPipeline(
        {
            "missing_rate": FeatureEliminationMissingRate(0.1),
            "cov": FeatureEliminationCoV(0.8),
            "kurtosis": FeatureEliminationKurtosis(3),
            "pearson": FeatureEliminationPearsonCorr(0.7),
            "vif": FeatureEliminationVIF(8),
        }
    )

    eod_feature_matrix = feature_elimination_pipeline.run(eod_feature_matrix, y=None)
    eod_feature_matrix = FeatureImputer(eod_feature_matrix).run()

    training_features = eod_feature_matrix.copy()

    # Save the data
    features_file_name = output_files["training_features"]
    training_features.to_csv(
        f"{output_bucket_path}{features_file_name}.csv",
        index=True,
    )

    # Metadata
    training_features_output.metadata["num_rows"] = len(training_features)
    training_features_output.metadata["num_columns"] = len(training_features.columns)
    training_features_output.metadata["column_names"] = list(training_features.columns)


# Compile the component
COMPONENT_FILE = f"pipelines/{PIPELINE_NAME}/components/{COMPONENT_NAME}.yaml"
print(f"Compiling {COMPONENT_FILE}")
compiler.Compiler().compile(
    feature_engineering_component,
    COMPONENT_FILE,
)
