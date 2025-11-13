from __future__ import annotations
from typing import Optional, Union
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array, check_is_fitted
from mapie.regression import MapieRegressor
from mapie.classification import MapieClassifier
from mapie.conformity_scores import BaseRegressionScore, BaseClassificationScore
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels


class MapieClassifierWithUncertainty(ClassifierMixin, BaseEstimator):
    """MAPIE Classifier with uncertainty compatible with predict/predict_proba"""

    def __init__(
        self,
        estimator: Optional[ClassifierMixin | BaseEstimator] = None,
        method: Optional[str] = None,
        n_folds: int = 5,
        test_size: Optional[int | float] = None,
        n_jobs: Optional[int] = None,
        conformity_score: Optional[BaseClassificationScore] = None,
        random_state: Optional[int | np.random.RandomState] = None,
        verbose: int = 0,
        mapie_alpha: float = 0.05,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.n_folds = n_folds
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.conformity_score = conformity_score
        self.random_state = random_state
        self.verbose = verbose
        self.cv = StratifiedKFold(
            n_splits=self.n_folds, random_state=random_state, shuffle=True
        )
        self.mapie_alpha = mapie_alpha
        self.mapie_classifier = MapieClassifier(
            estimator=estimator,
            method=method,
            test_size=test_size,
            n_jobs=n_jobs,
            conformity_score=conformity_score,
            random_state=random_state,
            cv=self.cv,
            verbose=verbose,
        )

    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        self.sample_weight_ = sample_weight
        self.mapie_classifier.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.mapie_classifier.estimator_.predict(X, "mean")

    def predict_uncert(self, X):
        """Calculate a refined uncertainty score for binary classification predictions using MAPIE confidence sets

        The approach refines uncertainty by combining base uncertainty with prediction variability via the following:
        1.) Estimation of out-of-fold (OOF) probabilities computed for each estimator in the ensemble
        2.) Standard deviation of OOF probabilities across ensemble computed as variability across ensembles
        3.) Base uncertainty derived via MAPIE confidence sets (empty sets also treated as highest uncertainty)
        4.) Refined uncertainty computed via base uncertainty and OOF standard deviation
        """

        check_is_fitted(self)
        X = check_array(X)
        y_pred_proba_k = np.asarray(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self.mapie_classifier.estimator_._predict_proba_oof_estimator)(
                    estimator,
                    X,
                )
                for estimator in self.mapie_classifier.estimator_.estimators_
            )
        )
        if y_pred_proba_k.shape[2] != 2:
            raise ValueError("predict_uncert is designed for binary classification")
        y_pred_std_k = y_pred_proba_k.std(axis=0)[:, 0]
        _, y_pi_mapie = self.mapie_classifier.predict(X, alpha=self.mapie_alpha)
        n_classes = y_pi_mapie.shape[1]
        confidence_set = y_pi_mapie[:, :, 0].sum(axis=1)
        confidence_set[confidence_set == 0] = n_classes
        base_uncertainty = confidence_set / n_classes
        refined_uncertainty = base_uncertainty * (1 - y_pred_std_k)
        return refined_uncertainty

    def predict_sets(self, X):
        """Return conformal prediction sets"""
        check_is_fitted(self)
        X = check_array(X)
        _, y_pi_mapie = self.mapie_classifier.predict(X, alpha=self.mapie_alpha)
        return y_pi_mapie[:, :, 0]

    def __getattr__(self, attr):
        return getattr(self.mapie_classifier, attr)

    def __str__(self):
        do_not_print = [
            "estimators_",
        ]
        attributes = [
            f"{key}='{value}'"
            for key, value in self.__dict__.items()
            if key not in do_not_print
        ]
        return f"MapieClassifierWithUncertainty({', '.join(attributes)})"


class MapieRegressorWithUncertainty(MapieRegressor):
    """
    Customised sklearn MapieRegressor with uncertainty
    """

    def __init__(
        self,
        estimator: Optional[RegressorMixin | BaseEstimator] = None,
        method: str = "plus",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        test_size: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[BaseRegressionScore] = None,
        mapie_alpha: float = 0.05,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.agg_function = agg_function
        self.verbose = verbose
        self.conformity_score = conformity_score
        self.random_state = random_state
        self.mapie_alpha = mapie_alpha

    def predict_uncert(self, X):
        """Allow uncertainties for Mapie"""
        predictions = self.predict(X, alpha=self.mapie_alpha)
        # return the difference (margin) between the lower and upper predictions
        return np.diff(predictions[1], axis=1).flatten()
