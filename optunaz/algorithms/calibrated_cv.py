import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import copy
from numbers import Integral

import sklearn
from sklearn.calibration import (
    LabelEncoder,
    label_binarize,
    IsotonicRegression,
    _SigmoidCalibration,
    CalibratedClassifierCV,
    _CalibratedClassifier,
)
from sklearn.utils import column_or_1d, indexable
from sklearn.utils.validation import check_is_fitted, _num_samples
from sklearn.utils._param_validation import (
    HasMethods,
    StrOptions,
)
from sklearn.model_selection import StratifiedKFold
import optunaz.algorithms.VennABERS as va


class _VennAbersCalibration(BaseEstimator, RegressorMixin):
    """
    VennABERS based on the implementation in VennABERS.py by Paolo Toccaceli, Royal Holloway, Univ. of London.
    Implementation based on "Large-scale probabilistic prediction with and without validity guarantees" (2015).
    See https://github.com/ptocca/VennABERS  for details.
    """

    def __init__(self, inference="proba"):
        self.inference = inference

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data."""
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        predictions = column_or_1d(X)
        y = column_or_1d(y)

        self.a_, self.b_ = predictions, y
        return self

    def predict(self, T):
        """Predict by ScoresToMultiProbs from VennABERS"""
        T = column_or_1d(T)

        calpoints = zip(self.a_, self.b_)
        vp = np.zeros(_num_samples(T))
        vp_0, vp_1 = va.ScoresToMultiProbs(calpoints, T)
        for i in range(_num_samples(T)):
            if self.inference == "proba":
                vp[i] = vp_1[i] / ((1.0 - vp_0[i]) + vp_1[i])
            elif self.inference == "uncert":
                vp[i] = vp_1[i] - vp_0[i]
        return vp


def _fit_calibrator_with_va(clf, predictions, y, classes, method, sample_weight=None):
    """
    Modification to fit_calibrator Scikit-learn function to add VennABERS
    """

    Y = label_binarize(y, classes=classes)
    label_encoder = LabelEncoder().fit(classes)
    pos_class_indices = label_encoder.transform(clf.classes_)
    calibrators = []
    for class_idx, this_pred in zip(pos_class_indices, predictions.T):
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
        elif method == "sigmoid":
            calibrator = _SigmoidCalibration()
        else:  # vennabers
            calibrator = _VennAbersCalibration()
        calibrator.fit(this_pred, Y[:, class_idx], sample_weight)
        calibrators.append(calibrator)

    pipeline = _CalibratedClassifier(clf, calibrators, method=method, classes=classes)
    return pipeline


# monkey patch the _fit_calibrator with _fit_calibrator_with_va here to ensure persistence
sklearn.calibration._fit_calibrator = _fit_calibrator_with_va


class CalibratedClassifierCVWithVA(CalibratedClassifierCV):
    """
    Customised sklearn CalibratedClassifierCV with VennABERS
    """

    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
            None,
        ],
        "method": [StrOptions({"isotonic", "sigmoid", "vennabers"})],  # add vennabers
        "cv": ["cv_object", StrOptions({"prefit"})],
        "n_jobs": [Integral, None],
        "ensemble": ["boolean"],
    }

    def __init__(
        self,
        estimator=None,
        *,
        method="sigmoid",
        n_folds=None,
        n_jobs=-1,
        ensemble=True,
        seed=42,
    ):
        self.estimator = estimator
        self.method = method
        self.n_folds = n_folds
        if hasattr(estimator, "num_workers"):
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.seed = seed
        self.cv = StratifiedKFold(
            n_splits=self.n_folds, random_state=self.seed, shuffle=True
        )

    def predict_uncert(self, X):
        """Provide uncertainties based on the VennABERS interval. Based on the concept of p0, p1 discordance
        proposed in "Comparison of Scaling Methods to Obtain Calibrated Probabilities of Activity for Protein-Ligand
        Predictions." J Chem Inf Model. (2020)
        """
        if self.method != "vennabers":
            raise AttributeError(
                "Uncertainty estimation only available for VennABERS method"
            )
        check_is_fitted(self)
        mean_proba = np.zeros((_num_samples(X), 1))
        for calibrated_classifier in self.calibrated_classifiers_:
            unc_calibrated_classifier = copy.deepcopy(calibrated_classifier)
            for cal in unc_calibrated_classifier.calibrators:
                cal.inference = "uncert"
            proba = unc_calibrated_classifier.predict_proba(X)[:, 1]
            mean_proba += proba.reshape(len(X), 1)

        mean_proba /= len(self.calibrated_classifiers_)
        return mean_proba.reshape(_num_samples(X))
