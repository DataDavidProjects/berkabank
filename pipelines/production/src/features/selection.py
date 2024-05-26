from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd
from probatus.feature_elimination import ShapRFECV
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class FeatureEliminationShap:
    """Feature elimination class.

    Attributes:
        model (Union[BaseEstimator, RandomizedSearchCV]): model to use for feature elimination
        step (float): step for feature elimination
        cv (int): number of cross-validation folds
        scoring (str): scoring metric
        n_jobs (int): number of parallel jobs
        standard_error_threshold (float): standard error threshold
        return_type (str): return type
        num_features (Union[int, str]): number of features to return

    Methods:
        run(X, y): fit the model


    Returns:
        list: reduced feature set
    """

    model: Union[BaseEstimator, RandomizedSearchCV]
    step: float = 0.2
    cv: int = 10
    scoring: str = "roc_auc"
    n_jobs: int = -1
    standard_error_threshold: float = 0.5
    return_type: str = "feature_names"
    num_features: Union[int, str] = "best"

    def run(self, X: pd.DataFrame, y: np.array) -> pd.DataFrame:
        """Run the feature elimination process.

        Args:
            X (pd.DataFrame): input features
            y (np.array): target variable

        Returns:
            list: reduced feature set
        """
        shap_elimination = ShapRFECV(
            model=self.model,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )

        grid_search = shap_elimination.fit(X, y)

        return grid_search.get_reduced_features_set(
            num_features=self.num_features,
            standard_error_threshold=self.standard_error_threshold,
            return_type=self.return_type,
        )
