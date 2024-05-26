import pandas as pd
from dataclasses import dataclass


@dataclass
class CoreBuilderTraining:
    accounts: pd.DataFrame
    eod_balance_agg: pd.DataFrame
    incidents: pd.DataFrame
    column_mapping: dict

    def __post_init__(self):
        self.core = None
        self.training_core = None
        self.eod_balance_training = None
        self.preprocess_data()

    def preprocess_data(self):
        self.eod_balance_agg[self.column_mapping["balance_date"]] = pd.to_datetime(
            self.eod_balance_agg[self.column_mapping["balance_date"]]
        )
        self.eod_balance_agg = self.eod_balance_agg.drop(
            columns=[
                self.column_mapping["target"],
                self.column_mapping["is_primary"],
                self.column_mapping["account_creation_date"],
                self.column_mapping["streak_id"],
                self.column_mapping["low_balance_flag"],
            ]
        )

    def build_core(self) -> pd.DataFrame:
        core = self.accounts.merge(
            self.incidents,
            how="left",
            on=[self.column_mapping["account_id"], self.column_mapping["district_id"]],
        ).sort_values(by=self.column_mapping["account_creation_date"])
        core[self.column_mapping["target"]] = ~core[
            self.column_mapping["incident_date"]
        ].isnull()
        core = core.loc[
            :, [self.column_mapping["account_id"], self.column_mapping["target"]]
        ]
        self.core = core
        return self.core

    def adjust_fair_core(self) -> pd.DataFrame:
        training_accounts = pd.DataFrame(
            self.eod_balance_agg[self.column_mapping["account_id"]].unique(),
            columns=[self.column_mapping["account_id"]],
        )
        self.training_core = self.core.merge(
            training_accounts, on=self.column_mapping["account_id"]
        )
        return self.training_core

    def build_eod_balance_training(self) -> pd.DataFrame:
        self.eod_balance_training = self.training_core.loc[
            :, [self.column_mapping["account_id"]]
        ].merge(self.eod_balance_agg, on=self.column_mapping["account_id"])
        return self.eod_balance_training

    def run(self) -> pd.DataFrame:
        self.build_core()
        self.adjust_fair_core()
        self.build_eod_balance_training()
        return {
            "eod_balance_training": self.eod_balance_training,
            "training_core": self.training_core,
        }
