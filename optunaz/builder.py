import numpy as np

from optunaz.algorithms.chem_prop import BaseChemProp
from optunaz.algorithms.fast_prop import BaseFastProp
from optunaz.config.build_from_opt import check_invalid_descriptor_param
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import ModelMode
from optunaz.descriptors import combine_covariates, descriptor_from_config
from optunaz.evaluate import get_merged_train_score, get_train_test_scores
from optunaz.utils import remove_failed_idx


def build(
    buildconfig: BuildConfig, merge_train_and_test_data: bool = False, cache=None
):
    """Build regressor or classifier model and return it."""

    valid_descriptors = check_invalid_descriptor_param(buildconfig.algorithm)
    if type(buildconfig.descriptor) not in valid_descriptors:
        raise ValueError("Build config does not have valid descriptor-algorithm pair")

    estimator = buildconfig.algorithm.estimator()
    if merge_train_and_test_data:
        train_smiles, train_y, train_aux = buildconfig.data.get_merged_sets()
        test_smiles, test_y, test_aux, test_X = None, None, None, None
    else:
        (
            train_smiles,
            train_y,
            train_aux,
            test_smiles,
            test_y,
            test_aux,
        ) = buildconfig.data.get_sets()
        if test_smiles:
            test_X, failed_idx = descriptor_from_config(
                test_smiles, buildconfig.descriptor, cache=cache
            )
            test_y, test_smiles, test_aux = remove_failed_idx(
                failed_idx, test_y, test_smiles, test_aux
            )
            if test_aux is not None:
                test_X = combine_covariates(buildconfig.descriptor, test_X, test_aux)
        else:
            test_X = None

    train_X, failed_idx = descriptor_from_config(
        train_smiles, buildconfig.descriptor, cache=cache
    )
    train_y, train_smiles, train_aux = remove_failed_idx(
        failed_idx, train_y, train_smiles, train_aux
    )
    if train_aux is not None:
        train_X = combine_covariates(buildconfig.descriptor, train_X, train_aux)

    estimator.fit(train_X, train_y)
    # get attr logic ensures that we do not overwrite existing attributes for ChemProp models
    for attr, value in [
        ("train_smiles_", train_smiles),
        ("test_smiles_", test_smiles),
        ("X_", train_X),
        ("test_X_", test_X),
        ("y_", train_y),
        ("test_y_", test_y),
        ("aux_", train_aux),
        ("test_aux_", test_aux),
    ]:
        if not hasattr(estimator, attr) or getattr(estimator, attr) is None:
            setattr(estimator, attr, value)

    estimator.test_pred, estimator.test_unc, estimator.test_err = None, None, None

    if isinstance(estimator, (BaseFastProp, BaseChemProp)):
        train_pred, train_unc = estimator.predict_uncert(train_X)
        test_pred, test_unc = (
            estimator.predict_uncert(test_X) if test_X is not None else (None, None)
        )
    else:
        if buildconfig.settings.mode == ModelMode.CLASSIFICATION:
            train_pred = estimator.predict_proba(train_X)[:, 1]
            test_pred = (
                estimator.predict_proba(test_X)[:, 1] if test_X is not None else None
            )
        else:  # ModelMode.REGRESSION
            train_pred = estimator.predict(train_X)
            test_pred = estimator.predict(test_X) if test_X is not None else None
        train_unc = (
            estimator.predict_uncert(train_X)
            if hasattr(estimator, "predict_uncert")
            else None
        )
        test_unc = (
            estimator.predict_uncert(test_X)
            if test_X is not None and hasattr(estimator, "predict_uncert")
            else None
        )

    if train_pred.shape != train_y.shape:
        estimator.train_pred, estimator.train_unc = train_pred[:, 0], train_unc[:, 0]
    if test_X is not None and test_pred.shape != test_y.shape:
        estimator.test_pred, estimator.test_unc = test_pred[:, 0], test_unc[:, 0]
    estimator.train_pred, estimator.train_unc = train_pred, train_unc
    estimator.train_err = abs(estimator.train_pred - train_y)
    if test_X is not None:
        estimator.test_pred, estimator.test_unc = test_pred, test_unc
        estimator.test_err = abs(estimator.test_pred - test_y)

    if (
        not merge_train_and_test_data
        and test_smiles is not None
        and len(test_smiles) >= 1
    ):
        train_scores, test_scores = get_train_test_scores(estimator, buildconfig)
    else:
        train_scores, test_scores = get_merged_train_score(estimator, buildconfig), None

    return estimator, train_scores, test_scores
