import pandas as pd
import numpy as np
import pytest


# Define path to files
path_to_files = "../berkabank/primary/"


# # Load the data
# files = ["accounts", "transactions"]
# data = {file: pd.read_csv(f"{path_to_files}{file}.csv") for file in files}

# # Convert dates to datetime
# data = {
#     key: df.apply(lambda col: pd.to_datetime(col) if "date" in col.name else col)
#     for key, df in data.items()
# }


def calculate_eod_balance(data: dict):
    # Generate list of all dates and account IDs
    dates = pd.date_range(
        start=data["transactions"]["transaction_date"].min(),
        end=data["transactions"]["transaction_date"].max(),
        freq="D",
    )
    all_account_ids = data["transactions"]["account_id"].unique()
    all_dates_df = pd.DataFrame(
        [(account_id, date) for account_id in all_account_ids for date in dates],
        columns=["account_id", "transaction_date"],
    )

    # Left join with transactions DataFrame
    eod_balance = pd.merge(
        all_dates_df,
        data["transactions"],
        how="left",
        on=["account_id", "transaction_date"],
    )

    # Sort eod_balance by account_id and transaction_date
    eod_balance = eod_balance.sort_values(["account_id", "transaction_date"])

    # Merge eod_balance with data['accounts'] on account_id to get account_creation_date
    eod_balance = eod_balance.merge(
        data["accounts"][["account_id", "account_creation_date"]],
        on="account_id",
        how="left",
    )

    # Remove transactions before account creation date
    eod_balance = eod_balance[
        eod_balance["transaction_date"] >= eod_balance["account_creation_date"]
    ]

    # Fill NaNs with 0
    eod_balance["transaction_amount"] = eod_balance["transaction_amount"].fillna(0)

    # Give sign to transaction amount based on transaction type
    eod_balance["daily_amount_flow"] = np.where(
        eod_balance["transaction_type"] == "outflow",
        -eod_balance["transaction_amount"],
        eod_balance["transaction_amount"],
    )

    # Calculate end-of-day balance
    eod_balance["end_of_day_balance"] = eod_balance.groupby("account_id")[
        "daily_amount_flow"
    ].cumsum()

    # Rename columns
    eod_balance.rename(columns={"transaction_date": "balance_date"}, inplace=True)

    # Sort by balance_date and account_id
    eod_balance = eod_balance.sort_values(["balance_date", "account_id"])

    # Filter columns
    eod_balance = eod_balance[
        [
            "account_id",
            "balance_date",
            "end_of_day_balance",
            "daily_amount_flow",
            "account_creation_date",
        ]
    ]

    return eod_balance


# Import the function to be tested


# Define a test case
def test_calculate_eod_balance():
    # Create sample data
    transactions = pd.DataFrame(
        {
            "account_id": [1, 1, 2, 2, 3, 3],
            "transaction_date": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-01",
                    "2022-01-02",
                ]
            ),
            "transaction_amount": [100, 50, 200, 100, 300, 150],
            "transaction_type": [
                "inflow",
                "outflow",
                "inflow",
                "outflow",
                "inflow",
                "outflow",
            ],
        }
    )
    accounts = pd.DataFrame(
        {
            "account_id": [1, 2, 3],
            "account_creation_date": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-01-01"]
            ),
        }
    )

    # Call the function to calculate EOD balance
    eod_balance = calculate_eod_balance(
        data={"transactions": transactions, "accounts": accounts}
    ).sort_values(["account_id", "balance_date"])
    # Reset the index of the DataFrame
    eod_balance = eod_balance.reset_index(drop=True)

    print("Function: calculate_eod_balance")
    print(eod_balance)

    # Define the expected output
    expected_output = pd.DataFrame(
        {
            "account_id": [1, 1, 2, 2, 3, 3],
            "balance_date": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-01",
                    "2022-01-02",
                ]
            ),
            "end_of_day_balance": [100, 50, 200, 100, 300, 150],
            "daily_amount_flow": [100, -50, 200, -100, 300, -150],
            "account_creation_date": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-01",
                    "2022-01-01",
                    "2022-01-01",
                    "2022-01-01",
                    "2022-01-01",
                ]
            ),
        }
    )
    print("Expected Output:")
    print(expected_output)

    # Compare the actual output with the expected output
    pd.testing.assert_frame_equal(eod_balance, expected_output)
