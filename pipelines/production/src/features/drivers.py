from dataclasses import dataclass
import pandas as pd
import numpy as np
import re
from typing import List, Union, Dict, Any

import warnings
from src.features.utils import *


# Suppress the PerformanceWarning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass
class EODBDrivers:
    eod_balance: pd.DataFrame
    drivers: List[str]

    def compute_drivers(self):
        ratio_drivers = [col for col in self.drivers if "ratio" in col]
        basic_drivers = [col for col in self.drivers if "ratio" not in col]
        ratio_num = [col.split("_ratio_")[0] for col in ratio_drivers]
        ratio_den = [col.split("_ratio_")[1] for col in ratio_drivers]

        normalized_drivers = basic_drivers + ratio_num + ratio_den
        print("----- Running EODBDrivers...")

        driver_matrix = pd.concat(
            [compute_driver_basic(self.eod_balance, x) for x in normalized_drivers],
            axis=1,
        )
        driver_matrix = driver_matrix.loc[:, ~driver_matrix.columns.duplicated()]

        eps = np.finfo(float).eps
        for feature in ratio_drivers:
            num, den = feature.split("_ratio_")
            driver_matrix[feature] = driver_matrix[num] / (driver_matrix[den] + eps)

        driver_matrix = driver_matrix.loc[:, self.drivers]

        return driver_matrix

    def run(self):
        driver_matrix = self.compute_drivers()
        print("----- EODBDrivers completed.")
        return driver_matrix
