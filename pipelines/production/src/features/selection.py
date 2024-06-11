from dataclasses import dataclass
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from probatus.feature_elimination import ShapRFECV
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


def drop_missing_percentage(df, threshold=0.1):
    """
    Returns a list of column names from the DataFrame where less than 10% of the data is missing.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.

    Returns:
    list: A list of column names.
    """
    columns = (
        (df.isna().mean() < threshold)
        .replace({True: 1, False: np.nan})
        .dropna()
        .index.to_list()
    )
    return df.loc[:, columns]


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
    check_additivity: bool = False
    min_features_to_select: int = 5

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
            min_features_to_select=self.min_features_to_select,
        )

        grid_search = shap_elimination.fit(
            X,
            y,
            check_additivity=self.check_additivity,
        )

        return grid_search.get_reduced_features_set(
            num_features=self.num_features,
            standard_error_threshold=self.standard_error_threshold,
            return_type=self.return_type,
        )


@dataclass
class FeatureEliminationMissingRate:
    """Feature elimination class.

    Attributes:
        missing_rate_threshold (float): missing rate threshold

    Methods:
        run(X, y): fit the model

    Returns:
        list: reduced feature set
    """

    missing_rate_threshold: float

    def run(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run the feature elimination process.

        Args:
            X (pd.DataFrame): input features
            y (np.array): target variable

        Returns:
            list: reduced feature set
        """
        missing_rate = X.isnull().mean()
        features_to_keep = missing_rate[
            missing_rate < self.missing_rate_threshold
        ].index
        return X[features_to_keep]


@dataclass
class FeatureEliminationKurtosis:
    """Feature elimination class.

    Attributes:
        kurtosis_threshold (float): kurtosis threshold

    Methods:
        run(X, y): fit the model

    Returns:
        list: reduced feature set
    """

    kurtosis_threshold: float

    def run(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run the feature elimination process.

        Args:
            X (pd.DataFrame): input features

        Returns:
            list: reduced feature set
        """
        kurt = X.kurt()
        features_to_keep = kurt[kurt < self.kurtosis_threshold].index
        return X[features_to_keep]


@dataclass
class FeatureEliminationCoV:
    """Feature elimination class.

    Attributes:
        coeffvar_threshold (float): coefficient of variation threshold

    Methods:
        run(X, y): fit the model

    Returns:
        list: reduced feature set
    """

    coeffvar_threshold: float

    def run(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run the feature elimination process.

        Args:
            X (pd.DataFrame): input features
            y (np.array): target variable

        Returns:
            list: reduced feature set
        """
        CoeffVar = X.std() / X.mean()
        features_to_keep = CoeffVar[CoeffVar > self.coeffvar_threshold].index
        return X[features_to_keep]


@dataclass
class FeatureEliminationPearsonCorr:
    """Feature elimination class based on Pearson correlation.

    Attributes:
        pearson_threshold (float): Pearson correlation threshold

    Methods:
        run(X): fit the model

    Returns:
        DataFrame: reduced feature set
    """

    pearson_threshold: float = 0.4

    def run(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run the feature elimination process.

        Args:
            X (pd.DataFrame): input features

        Returns:
            DataFrame: reduced feature set
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the threshold
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.pearson_threshold)
        ]

        # Drop features
        X = X.drop(X[to_drop], axis=1)

        return X


@dataclass
class FeatureEliminationVIF:
    """Feature elimination class.

    Attributes:
        vif_threshold (float): VIF threshold

    Methods:
        run(X): fit the model

    Returns:
        DataFrame: reduced feature set
    """

    vif_threshold: float
    filler: float = -1

    def run(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run the feature elimination process.

        Args:
            X (pd.DataFrame): input features

        Returns:
            DataFrame: reduced feature set
        """
        # Calculate VIF
        vif = pd.DataFrame()
        vif["variables"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X.fillna(self.filler).values, i)
            for i in range(X.shape[1])
        ]

        # Select variables below the threshold
        features_to_keep = vif[vif["VIF"] < self.vif_threshold]["variables"]

        return X[features_to_keep]


@dataclass
class FeatureEliminationPipeline:
    steps: Dict[str, Any]

    def run(self, X: pd.DataFrame, y: np.array) -> pd.DataFrame:
        """Run the feature elimination process.

        Args:
            X (pd.DataFrame): input features
            y (np.array): target variable

        Returns:
            list: reduced feature set
        """
        for step in self.steps:
            if y is None:
                X = self.steps[step].run(X)
            else:
                X = self.steps[step].run(X, y)
        return X
