# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict


# Define path to files
path_to_files = "../berkabank/elaboration/"


# Load the data
files = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
data = {file: pd.read_csv(f"{path_to_files}{file}.csv") for file in files}

# Convert dates to datetime
data = {
    key: df.apply(lambda col: pd.to_datetime(col) if "date" in col.name else col)
    for key, df in data.items()
}


def test_joint_accounts():
    # Merge account and disp dataframes
    merged_df = data["account"].merge(data["disp"], on=["account_id"])

    # Group by client_id and count the number of account_ids
    joint_accounts = merged_df.groupby("client_id").agg({"account_id": ["count"]})

    # Check if any client has more than 1 account
    assert (
        (joint_accounts > 1).sum() == 0
    ).all(), "There are joint accounts in the data"
