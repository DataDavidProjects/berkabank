from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pydantic import BaseModel
import pytest
import pandas as pd


@dataclass
class EODBalancePreprocessing:
    transaction_usage_flag: int
    seniority_account_flag: int
    target_balance: int
    incident_duration_days: int
    eod_balance: pd.DataFrame
    column_mapping: Dict
    compute_target: bool

    def __post_init__(self):
        # Convert the columns to the correct data types
        self.eod_balance[self.column_mapping["balance_date"]] = pd.to_datetime(
            self.eod_balance[self.column_mapping["balance_date"]]
        )
        self.eod_balance[self.column_mapping["account_creation_date"]] = pd.to_datetime(
            self.eod_balance[self.column_mapping["account_creation_date"]]
        )

        # Sort the DataFrame
        self.eod_balance = self.eod_balance.sort_values(
            [self.column_mapping["account_id"], self.column_mapping["balance_date"]]
        )

    def compute_n_transactions(self) -> pd.DataFrame:
        # Compute the number of cumulative transactions over the period for non-zero transactions
        non_zero_transactions = self.eod_balance[
            self.eod_balance[self.column_mapping["daily_amount_flow"]] != 0
        ]
        non_zero_transactions[self.column_mapping["n_transactions"]] = (
            non_zero_transactions.groupby(self.column_mapping["account_id"]).cumcount()
        )

        # Join with the original DataFrame and forward fill to maintain the count on days with 0 transactions
        self.eod_balance = self.eod_balance.join(
            non_zero_transactions[[self.column_mapping["n_transactions"]]],
            rsuffix="_non_zero",
        )
        self.eod_balance[self.column_mapping["n_transactions"]] = (
            self.eod_balance[self.column_mapping["n_transactions"]]
            .fillna(method="ffill")
            .fillna(0)
        )

        return self.eod_balance

    def calculate_seniority(self) -> pd.DataFrame:

        self.eod_balance = self.compute_n_transactions()

        # Calculate seniority as the number of days since account creation
        self.eod_balance["days_since_account_creation"] = (
            self.eod_balance[self.column_mapping["balance_date"]]
            - self.eod_balance[self.column_mapping["account_creation_date"]]
        ).dt.days
        return self.eod_balance

    def calculate_primary_flag(self) -> pd.DataFrame:
        # Each client is non primary by default, we will update this value by the time when requirements are met
        self.eod_balance[self.column_mapping["is_primary"]] = False
        self.eod_balance.loc[
            (
                self.eod_balance[self.column_mapping["n_transactions"]]
                >= self.transaction_usage_flag
            )
            & (
                self.eod_balance["days_since_account_creation"]
                >= self.seniority_account_flag
            ),
            self.column_mapping["is_primary"],
        ] = True
        return self.eod_balance

    def calculate_balance_incidents(self) -> pd.DataFrame:
        # Sort the DataFrame
        self.eod_balance = self.eod_balance.sort_values(
            [self.column_mapping["account_id"], self.column_mapping["balance_date"]]
        )
        # Create 'low_balance_flag'
        self.eod_balance[self.column_mapping["low_balance_flag"]] = (
            self.eod_balance[self.column_mapping["end_of_day_balance"]]
            < self.target_balance
        )
        # Create 'streak_id'
        self.eod_balance[self.column_mapping["streak_id"]] = (
            self.eod_balance[self.column_mapping["low_balance_flag"]]
            != self.eod_balance.groupby(self.column_mapping["account_id"])[
                self.column_mapping["low_balance_flag"]
            ].shift()
        ).cumsum()
        # Create 'low_balance_streak'
        self.eod_balance[self.column_mapping["low_balance_streak"]] = (
            self.eod_balance.groupby(
                [self.column_mapping["account_id"], self.column_mapping["streak_id"]]
            )[self.column_mapping["low_balance_flag"]].cumsum()
        )
        return self.eod_balance

    def calculate_target(self) -> pd.DataFrame:
        self.eod_balance[self.column_mapping["target"]] = (
            self.eod_balance[self.column_mapping["low_balance_streak"]]
            >= self.incident_duration_days
        ) & (self.eod_balance[self.column_mapping["is_primary"]])
        return self.eod_balance

    def run(self) -> pd.DataFrame:
        self.calculate_seniority()
        self.calculate_primary_flag()
        self.calculate_balance_incidents()
        if self.compute_target:
            self.calculate_target()
        return self.eod_balance


def test_calculate_target():
    # Define sample data
    eod_balance = pd.DataFrame(
        {
            "account_id": ["A1", "A1", "A1", "A1", "A1"],
            "balance_date": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"]
            ),
            "account_creation_date": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"]
            ),
            "end_of_day_balance": [10, 20, 30, 40, 50],
            "low_balance_streak": [1, 2, 3, 20, 21],
            "is_primary": [True, True, True, True, True],
            "daily_amount_flow": [0, 0, 0, 0, 0],
        }
    )
    column_mapping = {
        "account_id": "account_id",
        "balance_date": "balance_date",
        "account_creation_date": "account_creation_date",
        "end_of_day_balance": "end_of_day_balance",
        "low_balance_streak": "low_balance_streak",
        "is_primary": "is_primary",
        "daily_amount_flow": "daily_amount_flow",
        "n_transactions": "n_transactions",
        "target": "target",
    }

    # Create an instance of EODBalancePreprocessing
    bp = EODBalancePreprocessing(
        transaction_usage_flag=1,
        seniority_account_flag=1,
        target_balance=1,
        incident_duration_days=20,
        eod_balance=eod_balance,
        column_mapping=column_mapping,
        compute_target=True,
    )

    # Call the calculate_target method
    bp.calculate_target()

    # Define expected output
    expected_output = pd.DataFrame(
        {
            "account_id": ["A1", "A1", "A1", "A1", "A1"],
            "balance_date": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"]
            ),
            "account_creation_date": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"]
            ),
            "end_of_day_balance": [10, 20, 30, 40, 50],
            "low_balance_streak": [1, 2, 3, 20, 21],
            "is_primary": [True, True, True, True, True],
            "daily_amount_flow": [0, 0, 0, 0, 0],
            "target": [False, False, False, True, True],
        }
    )

    # Compare the actual output with the expected output
    pd.testing.assert_frame_equal(bp.eod_balance, expected_output)

    # Check if there is any target True with is_primary_account False
    assert bp.eod_balance[
        bp.eod_balance["target"] & ~bp.eod_balance["is_primary"]
    ].empty
