from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pydantic import BaseModel


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


@dataclass
class EodBalanceAggregation:
    eod_balance: pd.DataFrame
    accounts: pd.DataFrame
    incident_duration_days: int
    off_set_period_days: int
    column_mapping: dict

    def __post_init__(self):
        # Convert to datetime if not already
        self.eod_balance[self.column_mapping["balance_date"]] = pd.to_datetime(
            self.eod_balance[self.column_mapping["balance_date"]]
        )
        # Concat district information
        self.eod_balance = self.eod_balance.merge(
            self.accounts.loc[
                :,
                [self.column_mapping["account_id"], self.column_mapping["district_id"]],
            ],
            on=self.column_mapping["account_id"],
        )
        # Create incidents
        self.incidents = self.create_incidents()

    def create_incidents(self):
        # Collect incidents: Primary Account that stays for 20 consecutive days with eod_balance under amount.
        incidents = (
            self.eod_balance.loc[
                self.eod_balance[self.column_mapping["target"]] == True
            ]
            # Drop duplicates to keep only the first incident date
            .drop_duplicates(subset=self.column_mapping["account_id"], keep="first")
            .loc[
                :,
                [
                    self.column_mapping["account_id"],
                    self.column_mapping["balance_date"],
                    self.column_mapping["district_id"],
                ],
            ]
            .sort_values(self.column_mapping["balance_date"])
        )
        # Rename the balance date as incident date
        incidents = incidents.rename(
            {self.column_mapping["balance_date"]: self.column_mapping["incident_date"]},
            axis=1,
        )
        # Sort dataframes by date
        incidents.sort_values(self.column_mapping["incident_date"], inplace=True)
        self.eod_balance.sort_values(self.column_mapping["balance_date"], inplace=True)

        # Calculate t0: 1year and 20 days before day of incident, and t1: 20 days before incident; for each incident date
        incidents[self.column_mapping["t0"]] = incidents[
            self.column_mapping["incident_date"]
        ] - pd.DateOffset(days=self.incident_duration_days + self.off_set_period_days)
        incidents[self.column_mapping["t1"]] = incidents[
            self.column_mapping["incident_date"]
        ] - pd.DateOffset(days=self.incident_duration_days)

        # Filter out incidents that are before the beginning of the dataset, and therefore not usable.
        incidents = incidents.loc[
            incidents[self.column_mapping["t0"]]
            > self.eod_balance[self.column_mapping["balance_date"]].min(),
            :,
        ]

        return incidents

    def collect_eod_balance_incidents_accounts(self):
        # Collect district_id information for each account id
        self.eod_balance = self.eod_balance.merge(
            self.accounts.loc[:, [self.column_mapping["account_id"]]],
            on=self.column_mapping["account_id"],
        )

        # Extend with information of incident, connect to each period to t0 and t1
        eod_balance_incidents = pd.merge(
            self.eod_balance,
            self.incidents,
            on=[self.column_mapping["account_id"], self.column_mapping["district_id"]],
        )
        # Filter period between t0 and t1 for each account_id in incidents
        eod_balance_period = eod_balance_incidents.loc[
            eod_balance_incidents[self.column_mapping["balance_date"]].between(
                eod_balance_incidents[self.column_mapping["t0"]],
                eod_balance_incidents[self.column_mapping["t1"]],
            )
        ]
        eod_balance_incidents = eod_balance_period.sort_values(
            [self.column_mapping["balance_date"], self.column_mapping["account_id"]]
        )

        return eod_balance_incidents

    def fair_no_incidents_accounts(self):
        # Extend info about incidents and districts ( if account is in district id where incident was recorded )
        accounts_incident_flag = self.accounts.assign(
            incident_flag=self.accounts[self.column_mapping["account_id"]].isin(
                self.incidents[self.column_mapping["account_id"]]
            )
        )
        # Extend t0 and t1 on accounts_incident_flag to connect accounts of same district with different outcome.
        account_extended_info_incident = (
            accounts_incident_flag.merge(
                self.incidents,
                on=[
                    self.column_mapping["account_id"],
                    self.column_mapping["district_id"],
                ],
                how="left",
            )
            .sort_values(self.column_mapping["district_id"])
            .ffill()
            .dropna()
        )
        # Filter account id without an incident
        no_incident_accounts = account_extended_info_incident.loc[
            ~accounts_incident_flag[self.column_mapping["account_id"]].isin(
                self.incidents[self.column_mapping["account_id"]]
            ),
            :,
        ]
        # Declare linked incident date
        no_incident_accounts = no_incident_accounts.rename(
            {self.column_mapping["incident_date"]: "linked_incident_date"}, axis=1
        )
        return no_incident_accounts

    def collect_eof_balance_no_incidents_accounts(self):

        no_incident_accounts = self.fair_no_incidents_accounts()
        eod_balance_linked = self.eod_balance.merge(
            no_incident_accounts,
            on=[
                self.column_mapping["account_id"],
                self.column_mapping["district_id"],
                self.column_mapping["account_creation_date"],
            ],
        )
        # Filter relevant period if balance date between t0 and t1
        eod_balance_period_linked = eod_balance_linked.loc[
            eod_balance_linked[self.column_mapping["balance_date"]].between(
                eod_balance_linked[self.column_mapping["t0"]],
                eod_balance_linked[self.column_mapping["t1"]],
            )
        ]

        return eod_balance_period_linked

    def collect_fair_eod_balance(self):

        eod_balance_incidents = self.collect_eod_balance_incidents_accounts()
        eod_balance_period_linked = self.collect_eof_balance_no_incidents_accounts()

        fair_eod_balance_period = (
            pd.concat([eod_balance_period_linked, eod_balance_incidents], axis=0)
            .sort_values(self.column_mapping["balance_date"])
            .loc[
                :,
                [
                    self.column_mapping["account_id"],
                    self.column_mapping["balance_date"],
                    self.column_mapping["end_of_day_balance"],
                    self.column_mapping["daily_amount_flow"],
                    self.column_mapping["account_creation_date"],
                    self.column_mapping["n_transactions"],
                    self.column_mapping["days_since_account_creation"],
                    self.column_mapping["is_primary"],
                    self.column_mapping["low_balance_flag"],
                    self.column_mapping["streak_id"],
                    self.column_mapping["low_balance_streak"],
                    self.column_mapping["target"],
                    self.column_mapping["district_id"],
                ],
            ]
        )

        return fair_eod_balance_period

    def run(self):
        fair_eod_balance_period = self.collect_fair_eod_balance()
        return fair_eod_balance_period
