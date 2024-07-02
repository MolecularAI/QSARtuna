import numpy as np
from typing import Optional, Union
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from mapie.regression import MapieRegressor
from mapie.conformity_scores import ConformityScore


class MapieWithUncertainty(MapieRegressor):
    """
    Customised sklearn MapieRegressor with uncertainty
    """

    def __init__(
        self,
        estimator: Optional[RegressorMixin | BaseEstimator] = None,
        method: str = "plus",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[ConformityScore] = None,
        mapie_alpha: float = 0.05,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.agg_function = agg_function
        self.verbose = verbose
        self.conformity_score = conformity_score
        self.mapie_alpha = mapie_alpha

    def predict_uncert(self, X):
        """Allow uncertainties for Mapie"""
        predictions = self.predict(X, alpha=self.mapie_alpha)
        # return the difference (margin) between the lower and upper predictions
        return np.diff(predictions[1], axis=1).flatten()
