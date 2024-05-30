import os
from pathlib import Path
from kfp import compiler, dsl
from utils.vertexai import vertex_authenticate


# -------------------- Import components ---------------------------------------
from components.eod_balance.definition import create_eod_balance_component
from components.balance_processing.definition import process_eod_balance_component
from components.balance_aggregation.definition import eod_balance_aggregation_component
from components.incidents.definition import incidents_component
from components.core_training.definition import core_fair_eod_balance_training_component
from components.feature_engineering.definition import feature_engineering_component
from components.feature_selection.definition import feature_selection_component
from components.hyperparameter_tuning.definition import hyperparameter_tuning_component
from components.model_upload.definition import model_registry_component
from components.continuous_training.definition import training_trigger_component
from components.batch_prediction.definition import batch_prediction_component
import datetime
from config.pipeline import *

# Authenticate with Google Cloud SDK for Vertex AI
aiplatform_client = vertex_authenticate()
# ------------------------------------------------------------------------------


# --------------------- Compile Pipeline ---------------------------------------
@dsl.pipeline(
    pipeline_root=f"gs://{os.environ.get('BUCKET_NAME')}/{PIPELINE_NAME}/run/",
    name=PIPELINE_NAME,
    description=f"Pipeline on Vertex AI for {PIPELINE_NAME}",
)
def pipeline():

    # -------------------- Compile Components ----------------------------------
    create_eod_balance_task = create_eod_balance_component(**eod_balance_component_args)

    # Process EOD Balance
    preprocess_eod_balance_task = process_eod_balance_component(
        **preprocess_eod_balance_component_args,
        eod_balance=create_eod_balance_task.outputs["eod_balance_output"],
    )

    # Training Trigger
    training_trigger_task = training_trigger_component(
        **training_trigger_component_args
    )

    training_trigger_task.after(preprocess_eod_balance_task)

    # --------------------- Conditional Workflow  ------------------------------

    with dsl.If(
        name="Training Condition",
        condition=(
            training_trigger_task.outputs["trigger"]  # pylint: disable=no-member
            == "True"
        ),
    ) as training_job:
        # Create EOD Balance

        # Create incidents
        create_incidents_task = incidents_component(
            **incidents_component_args,
            eod_balance_preprocessed=preprocess_eod_balance_task.outputs[
                "eod_balance_preprocessed_output"
            ],
        )

        # Compute Aggregation for training - Fair EOD Balance Period
        aggregation_eod_balance_task = eod_balance_aggregation_component(
            **aggregation_eod_balance_component_args,
            eod_balance_preprocessed=preprocess_eod_balance_task.outputs[
                "eod_balance_preprocessed_output"
            ],
        )

        # Create EODB Training
        core_fair_eod_balance_training_task = core_fair_eod_balance_training_component(
            **eod_balance_training_args,
            eod_balance_aggregation=aggregation_eod_balance_task.outputs[
                "eod_balance_aggregation_output"
            ],
        )

        # Features Training
        features_engineering_task = feature_engineering_component(
            **features_enigneering_component_args,
            eod_balance_training=core_fair_eod_balance_training_task.outputs[
                "eod_balance_training_output"
            ],
            incidents=create_incidents_task.outputs["incidents_output"],
        )

        # Feature Selection
        feature_selection_task = feature_selection_component(
            **feature_selection_component_args,
            training_features_input=features_engineering_task.outputs[
                "training_features_output"
            ],
            core_training_input=core_fair_eod_balance_training_task.outputs[
                "core_training_output"
            ],
        )

        # Hyperparameter Tuning
        hyperparameter_tuning_task = hyperparameter_tuning_component(
            **hyperparameter_tuning_component_args,
            drivers_input=feature_selection_task.outputs["drivers_output"],
            training_drivers_input=feature_selection_task.outputs[
                "training_drivers_output"
            ],
            core_training_input=core_fair_eod_balance_training_task.outputs[
                "core_training_output"
            ],
        )

        # --------------------------------------------------------------------------

        # -------------------------- Conditional Workflow  -------------------------

        with dsl.If(
            name="Deployment Condition",
            condition=hyperparameter_tuning_task.outputs[  # pylint: disable=no-member
                "auc_validation"
            ]
            >= PROMOTION_PERFORMANCE,
        ):
            # Model Registry - Upload and Deployment
            model_upload_task = model_registry_component(
                **model_registry_component_args,
                best_model_input=hyperparameter_tuning_task.outputs[  # pylint: disable=no-member
                    "best_model_output"
                ],
            )
            model_upload_task.after(hyperparameter_tuning_task)
            batch_prediction_task = batch_prediction_component(
                **batch_prediction_component_args,
            )
            batch_prediction_task.after(model_upload_task)

            # ------------------------------------------------------------------

    with dsl.If(
        name="Fetch Model Condition",
        condition=training_trigger_task.outputs["trigger"] == "False",
    ):
        # Batch Prediction
        batch_prediction_task = batch_prediction_component(
            **batch_prediction_component_args,
        )
        batch_prediction_task.after(training_trigger_task)


# ------------------------------------------------------------------------------


# Version
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
time_based_version = f"_v_{timestamp}"
VERSION = time_based_version
# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=f"pipelines/{PIPELINE_NAME}/registry/{PIPELINE_NAME}_pipeline{VERSION}.json",
)
