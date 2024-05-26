from dataclasses import dataclass
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
import pandas as pd


@dataclass
class HyperparameterTuning:
    estimator: any
    params: dict
    scoring: str
    n_iter: int
    n_splits: int
    random_state: int

    def perform_search(self, X, y):
        model = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.params,
            scoring=self.scoring,
            cv=StratifiedShuffleSplit(
                n_splits=self.n_splits, test_size=0.1, random_state=42
            ),
            n_iter=self.n_iter,
            random_state=42,
            n_jobs=-1,
        )
        grid = model.fit(X, y)
        return grid

    def run(self, X, y):
        grid = self.perform_search(X, y)
        report = (
            pd.DataFrame(grid.cv_results_)
            .sort_values("rank_test_score")
            .loc[:, ["rank_test_score", "mean_test_score", "std_test_score", "params"]]
            .set_index("rank_test_score")
        )

        best = {
            "best_estimator": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
        }
        return {"best": best, "report": report}
