from typing import List
import numpy as np
from sklearn.metrics import check_scoring
from sklearn.calibration import calibration_curve
from optunaz import objective
from optunaz.config import ModelMode
from optunaz.config.buildconfig import BuildConfig


def get_scores(mode: ModelMode) -> List[str]:
    if mode == ModelMode.REGRESSION:
        scores = objective.regression_scores
    elif mode == ModelMode.CLASSIFICATION:
        scores = objective.classification_scores
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    return scores


def get_train_test_scores(
    estimator, buildconfig: BuildConfig, train_X, train_y, test_X, test_y
):
    scores = get_scores(buildconfig.settings.mode)
    train_scores = check_scoring(estimator, scoring=scores)(estimator, train_X, train_y)
    test_scores = check_scoring(estimator, scoring=scores)(estimator, test_X, test_y)
    return train_scores, test_scores


def get_merged_train_score(estimator, buildconfig: BuildConfig, train_X, train_y):
    scores = get_scores(buildconfig.settings.mode)
    train_scores = check_scoring(estimator, scoring=scores)(estimator, train_X, train_y)
    return train_scores


def calibration_analysis(y_test, y_pred):
    try:
        frac_true, frac_pred = calibration_curve(y_test, y_pred, n_bins=15)
        bin_edges = frac_pred
    except ValueError:
        # weight each bin by the total number of values so that the sum of all bars equal unity
        weights = np.ones_like(y_test) / len(y_test)
        # calculate fraction of true points across uniform bins
        frac_true, bin_edges = np.histogram(y_test, bins=15, weights=weights)
        # calculate fraction of pred points across uniform true bins
        frac_pred, _ = np.histogram(y_pred, bins=bin_edges, weights=weights)
        # convert to cumulative sum for plotting
        frac_true = np.cumsum(frac_true)
        frac_pred = np.cumsum(frac_pred)
    return list(zip(bin_edges, frac_true, frac_pred))
