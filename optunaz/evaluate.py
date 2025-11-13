from typing import List

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import check_scoring

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


def get_train_test_scores(estimator, buildconfig: BuildConfig):
    scores = get_scores(buildconfig.settings.mode)
    train_scores = check_scoring(estimator, scoring=scores)(
        estimator,
        estimator.X_,
        estimator.y_[:, 0]
        if hasattr(estimator.y_, "ndim") and estimator.y_.ndim > 1
        else estimator.y_,
    )
    test_scores = check_scoring(estimator, scoring=scores)(
        estimator,
        estimator.test_X_,
        estimator.test_y_[:, 0]
        if hasattr(estimator.test_y_, "ndim") and estimator.test_y_.ndim > 1
        else estimator.test_y_,
    )
    if estimator.train_unc is not None and estimator.test_unc is not None:
        train_uq_scores = uncertainty_assessment(
            estimator.train_err, estimator.train_unc
        )
        test_uq_scores = uncertainty_assessment(estimator.test_err, estimator.test_unc)
        train_scores.update(train_uq_scores)
        test_scores.update(test_uq_scores)
    train_scores = {k: float(v) for k, v in train_scores.items()}
    test_scores = {k: float(v) for k, v in test_scores.items()}

    return train_scores, test_scores


def get_merged_train_score(estimator, buildconfig: BuildConfig):
    scores = get_scores(buildconfig.settings.mode)
    train_scores = check_scoring(estimator, scoring=scores)(
        estimator,
        estimator.X_,
        estimator.y_[:, 0]
        if hasattr(estimator.y_, "ndim") and estimator.y_.ndim > 1
        else estimator.y_,
    )
    if estimator.train_unc is not None:
        train_uq_scores = uncertainty_assessment(
            estimator.train_err, estimator.train_unc
        )
        train_scores.update(train_uq_scores)
    train_scores = {k: float(v) for k, v in train_scores.items()}
    return train_scores


def calibration_analysis(y_test, y_pred, norm=False):
    try:
        frac_true, frac_pred = calibration_curve(y_test, y_pred, n_bins=15)
        bin_edges = frac_pred
    except ValueError:
        if norm:
            from sklearn.preprocessing import MinMaxScaler

            y_test = MinMaxScaler().fit_transform(y_test.reshape(-1, 1)).flatten()
            y_pred = MinMaxScaler().fit_transform(y_pred.reshape(-1, 1)).flatten()
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


def uncertainty_assessment(error, uncertainty):
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.preprocessing import StandardScaler

    from optunaz.metrics import concordance_index_err

    z_error = StandardScaler().fit_transform(error.reshape(-1, 1)).flatten()
    z_unc = StandardScaler().fit_transform(uncertainty.reshape(-1, 1)).flatten()

    scores = {
        "uncertainty_r2": r2_score(error, uncertainty),
        "uncertainty_r2_z": r2_score(z_error, z_unc),
        "uncertainty_corrcoef": np.corrcoef(error, uncertainty)[0, 1],
        "uncertainty_spearmanr": spearmanr(error, uncertainty)[0],
        "uncertainty_concordance_index": concordance_index_err(error, uncertainty),
        "uncertainty_cosine": cosine_similarity([error], [uncertainty])[0, 0],
        "uncertainty_cosine_z": cosine_similarity([z_error], [z_unc])[0, 0],
        "uncertainty_euclidean_dist": euclidean_distances([error], [uncertainty])[0, 0],
        "uncertainty_euclidean_dist_z": euclidean_distances([z_error], [z_unc])[0, 0],
    }
    return scores
