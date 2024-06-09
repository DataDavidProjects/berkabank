from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any, Callable
import pandas as pd
import re
import warnings
import random
import numpy as np
from typing import Callable


@dataclass
class FeatureImputer:
    df: pd.DataFrame
    filler: float = -1

    def run(self):
        print("--- Running FeatureImputer...")
        imputed_df = self.df.fillna(self.filler)
        print("--- FeatureImputer completed.")
        return imputed_df
