from dataclasses import dataclass
import pandas as pd


@dataclass
class EODBDrivers:
    eod_balance_preprocessed: pd.DataFrame
    column_mapping: dict

    def run(self):
        return self.eod_balance_preprocessed
