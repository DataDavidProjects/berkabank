from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pydantic import BaseModel


@dataclass
class EodBalanceBuilder:
    """
    Utility class for creating end-of-day balance from transactions and accounts data.
    Steps:
        1. Generate list of all dates and account IDs
        2. Left join with self.transactions DataFrame
        3. Sort eod_balance by account_id and transaction_date
        4. Merge eod_balance with account on account_id to get account_creation_date
        5. Remove self.transactions before account creation date
        6. Fill NaNs with 0
        7. Give sign to transaction amount based on transaction type
        8. Calculate end-of-day balance
        9. Rename columns
        10. Sort by balance_date and account_id
        11. Filter columns
        12. Return eod_balance

    Attributes:
        transactions: DataFrame of transactions
        accounts: DataFrame of accounts
        config: Dictionary of configuration parameters
    Methods:
        run: Creates end-of-day balance from transactions and accounts data.

    Returns:
        eod_balance: DataFrame of end-of-day balance
    """

    transactions: pd.DataFrame
    accounts: pd.DataFrame
    column_mapping: Dict[str, Optional[str]]

    def run(self):

        # Generate list of all dates and account IDs
        dates = pd.date_range(
            start=self.transactions[self.column_mapping["transaction_date"]].min(),
            end=self.transactions[self.column_mapping["transaction_date"]].max(),
            freq="D",
        )
        all_account_ids = self.transactions[self.column_mapping["account_id"]].unique()
        all_dates_df = pd.DataFrame(
            [(account_id, date) for account_id in all_account_ids for date in dates],
            columns=[
                self.column_mapping["account_id"],
                self.column_mapping["transaction_date"],
            ],
        )

        # Convert dates to datetime
        all_dates_df[self.column_mapping["transaction_date"]] = pd.to_datetime(
            all_dates_df[self.column_mapping["transaction_date"]]
        )
        self.accounts["account_creation_date"] = pd.to_datetime(
            self.accounts["account_creation_date"]
        )
        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"]
        )

        # Left join with self.transactions DataFrame
        eod_balance = pd.merge(
            all_dates_df,
            self.transactions,
            how="left",
            on=[
                self.column_mapping["account_id"],
                self.column_mapping["transaction_date"],
            ],
        )

        # Sort eod_balance by account_id and transaction_date
        eod_balance = eod_balance.sort_values(
            [self.column_mapping["account_id"], self.column_mapping["transaction_date"]]
        )

        # Merge eod_balance with account on account_id to get account_creation_date
        eod_balance = eod_balance.merge(
            self.accounts[
                [
                    self.column_mapping["account_id"],
                    self.column_mapping["account_creation_date"],
                ]
            ],
            on=self.column_mapping["account_id"],
            how="left",
        )

        # Remove self.transactions before account creation date
        eod_balance = eod_balance[
            eod_balance[self.column_mapping["transaction_date"]]
            >= eod_balance[self.column_mapping["account_creation_date"]]
        ]

        # Fill NaNs with 0
        eod_balance[self.column_mapping["transaction_amount"]] = eod_balance[
            self.column_mapping["transaction_amount"]
        ].fillna(0)

        # Give sign to transaction amount based on transaction type
        eod_balance[self.column_mapping["daily_amount_flow"]] = np.where(
            eod_balance[self.column_mapping["transaction_type"]]
            == self.column_mapping["outflow"],
            -eod_balance[self.column_mapping["transaction_amount"]],
            eod_balance[self.column_mapping["transaction_amount"]],
        )

        # Calculate end-of-day balance
        eod_balance[self.column_mapping["end_of_day_balance"]] = eod_balance.groupby(
            self.column_mapping["account_id"]
        )[self.column_mapping["daily_amount_flow"]].cumsum()

        eod_balance["flow_category"] = (
            eod_balance[self.column_mapping["daily_amount_flow"]]
            .gt(0)
            .replace({True: "inflow", False: "outflow"})
        )

        eod_balance["daily_amount_inflow"] = eod_balance[
            self.column_mapping["daily_amount_flow"]
        ].clip(lower=0)

        eod_balance["daily_amount_outflow"] = (
            eod_balance[self.column_mapping["daily_amount_flow"]].clip(upper=0).abs()
        )

        # Rename columns
        eod_balance.rename(
            columns={
                self.column_mapping["transaction_date"]: self.column_mapping[
                    "balance_date"
                ]
            },
            inplace=True,
        )

        # Sort by balance_date and account_id
        eod_balance = eod_balance.sort_values(
            [self.column_mapping["balance_date"], self.column_mapping["account_id"]]
        )

        # Filter columns
        eod_balance = eod_balance[
            [
                self.column_mapping["account_id"],
                self.column_mapping["balance_date"],
                self.column_mapping["end_of_day_balance"],
                self.column_mapping["daily_amount_flow"],
                self.column_mapping["account_creation_date"],
                "daily_amount_inflow",
                "daily_amount_outflow",
            ]
        ]
        return eod_balance
