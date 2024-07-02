import warnings
from inspect import signature
from sklearn.utils import Bunch
from sklearn.base import (
    _fit_context,
    clone,
)
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils import (
    _safe_indexing,
)
from sklearn.utils._response import _get_response_values, _process_predict_proba
from sklearn.utils.metadata_routing import (
    _routing_enabled,
    process_routing,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    _check_response_method,
    _check_sample_weight,
)


import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import copy
from numbers import Integral

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
        "method": [StrOptions({"isotonic", "sigmoid", "vennabers"})],  # add VennABERS
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
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.seed = seed
        self.cv = StratifiedKFold(
            n_splits=self.n_folds, random_state=self.seed, shuffle=True
        )

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None, **fit_params):
        check_classification_targets(y)
        X, y = indexable(X, y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        estimator = self._get_estimator()

        self.calibrated_classifiers_ = []
        if self.cv == "prefit":
            check_is_fitted(self.estimator, attributes=["classes_"])
            self.classes_ = self.estimator.classes_

            predictions, _ = _get_response_values(
                estimator,
                X,
                response_method=["decision_function", "predict_proba"],
            )
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)

            calibrated_classifier = _fit_calibrator_with_va(
                estimator,
                predictions,
                y,
                self.classes_,
                self.method,
                sample_weight,
            )  # add VennABERS
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            label_encoder_ = LabelEncoder().fit(y)
            self.classes_ = label_encoder_.classes_

            if _routing_enabled():
                routed_params = process_routing(
                    self,
                    "fit",
                    sample_weight=sample_weight,
                    **fit_params,
                )
            else:
                fit_parameters = signature(estimator.fit).parameters
                supports_sw = "sample_weight" in fit_parameters
                if sample_weight is not None and not supports_sw:
                    estimator_name = type(estimator).__name__
                    warnings.warn(
                        f"{estimator_name} does not appear to accept"
                        " sample_weight, sample weights will only be used for the"
                        " calibration itself. This can be caused by a limitation of"
                        " the current scikit-learn API. See the following issue for"
                        " more details:"
                        " https://github.com/scikit-learn/scikit-learn/issues/21134."
                        " Be warned that the result of the calibration is likely to be"
                        " incorrect."
                    )
                routed_params = Bunch()
                routed_params.splitter = Bunch(split={})  # no routing for splitter
                routed_params.estimator = Bunch(fit=fit_params)
                if sample_weight is not None and supports_sw:
                    routed_params.estimator.fit["sample_weight"] = sample_weight

            if isinstance(self.cv, int):
                n_folds = self.cv
            elif hasattr(self.cv, "n_splits"):
                n_folds = self.cv.n_splits
            else:
                n_folds = None
            if n_folds and np.any(
                [np.sum(y == class_) < n_folds for class_ in self.classes_]
            ):
                raise ValueError(
                    f"Requesting {n_folds}-fold "
                    "cross-validation but provided less than "
                    f"{n_folds} examples for at least one class."
                )
            cv = check_cv(self.cv, y, classifier=True)

            if self.ensemble:
                parallel = Parallel(n_jobs=self.n_jobs)
                self.calibrated_classifiers_ = parallel(
                    delayed(_fit_classifier_calibrator_pair_with_va)(
                        clone(estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        method=self.method,
                        classes=self.classes_,
                        sample_weight=sample_weight,
                        fit_params=routed_params.estimator.fit,
                    )  # add VennABERS
                    for train, test in cv.split(X, y, **routed_params.splitter.split)
                )
            else:
                this_estimator = clone(estimator)
                method_name = _check_response_method(
                    this_estimator,
                    ["decision_function", "predict_proba"],
                ).__name__
                predictions = cross_val_predict(
                    estimator=this_estimator,
                    X=X,
                    y=y,
                    cv=cv,
                    method=method_name,
                    n_jobs=self.n_jobs,
                    params=routed_params.estimator.fit,
                )
                if len(self.classes_) == 2:
                    # Ensure shape (n_samples, 1) in the binary case
                    if method_name == "predict_proba":
                        # Select the probability column of the postive class
                        predictions = _process_predict_proba(
                            y_pred=predictions,
                            target_type="binary",
                            classes=self.classes_,
                            pos_label=self.classes_[1],
                        )
                    predictions = predictions.reshape(-1, 1)

                this_estimator.fit(X, y, **routed_params.estimator.fit)
                calibrated_classifier = _fit_calibrator_with_va(
                    this_estimator,
                    predictions,
                    y,
                    self.classes_,
                    self.method,
                    sample_weight,
                )  # add VennABERS
                self.calibrated_classifiers_.append(calibrated_classifier)

        first_clf = self.calibrated_classifiers_[0].estimator
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_
        return self

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


def _fit_classifier_calibrator_pair_with_va(
    estimator,
    X,
    y,
    train,
    test,
    method,
    classes,
    sample_weight=None,
    fit_params=None,
):
    """
    Modification to _fit_classifier_calibrator_pair Scikit-learn function to add VennABERS
    """
    fit_params_train = _check_method_params(X, params=fit_params, indices=train)
    X_train, y_train = _safe_indexing(X, train), _safe_indexing(y, train)
    X_test, y_test = _safe_indexing(X, test), _safe_indexing(y, test)

    estimator.fit(X_train, y_train, **fit_params_train)

    predictions, _ = _get_response_values(
        estimator,
        X_test,
        response_method=["decision_function", "predict_proba"],
    )
    if predictions.ndim == 1:
        # Reshape binary output from `(n_samples,)` to `(n_samples, 1)`
        predictions = predictions.reshape(-1, 1)

    sw_test = None if sample_weight is None else _safe_indexing(sample_weight, test)
    calibrated_classifier = _fit_calibrator_with_va(
        estimator, predictions, y_test, classes, method, sample_weight=sw_test
    )
    return calibrated_classifier
