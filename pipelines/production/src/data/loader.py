import pandas as pd
import dataclasses
from google.cloud import storage
import os
import pickle
from kfp import dsl


@dataclasses.dataclass
class DataLoader:
    """Utility class for loading data from various sources.

    Attributes:
        config: _description_
        storage_client: _description_

    Methods:
        load_bucket: load data from a Cloud Storage bucket
        load_pickle: load data from a pickle file
        save_pickle: save data to a pickle file
    """

    config: dict = None
    storage_client: storage.Client = storage.Client()

    @staticmethod
    def load_bucket(path: str, files: list) -> pd.DataFrame:
        # Cloud Storage FUSE notation /gcs/ to access the data

        data_dict = {
            file: pd.read_csv(os.path.join(path, f"{file}.csv")) for file in files
        }

        return data_dict

    @staticmethod
    def load_pickle(artifact: dsl.Artifact) -> pd.DataFrame:
        with open(artifact.path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def save_pickle(artifact: dsl.Artifact, data: pd.DataFrame) -> None:
        with open(artifact.path, "wb") as file:
            pickle.dump(data, file)
