from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pydantic import BaseModel


@dataclass
class IncidentsBuilder:
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

    def run(self):
        return self.create_incidents()
