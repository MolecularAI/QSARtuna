import numpy as np
from optunaz.config.buildconfig import BuildConfig
from optunaz.descriptors import descriptor_from_config
from optunaz.evaluate import get_merged_train_score, get_train_test_scores
from optunaz.utils import remove_failed_idx
from optunaz.config.build_from_opt import check_invalid_descriptor_param


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
        if test_smiles is not None and len(test_smiles) > 0:
            test_X, failed_idx = descriptor_from_config(
                test_smiles, buildconfig.descriptor, cache=cache
            )
            test_y, test_smiles, test_aux = remove_failed_idx(
                failed_idx, test_y, test_smiles, test_aux
            )
            if test_aux is not None:
                test_X = np.hstack((test_X, test_aux))
        else:
            test_X = None

    train_X, failed_idx = descriptor_from_config(
        train_smiles, buildconfig.descriptor, cache=cache
    )
    train_y, train_smiles, train_aux = remove_failed_idx(
        failed_idx, train_y, train_smiles, train_aux
    )

    if train_aux is not None:
        train_X = np.hstack((train_X, train_aux))

    estimator.fit(train_X, train_y)
    estimator.train_smiles_ = train_smiles
    estimator.X_ = train_X
    estimator.y_ = train_y
    estimator.aux_ = train_aux
    estimator.test_smiles_ = test_smiles
    estimator.test_X_ = test_X
    estimator.test_y_ = test_y
    estimator.test_aux_ = test_aux

    if (
        not merge_train_and_test_data
        and test_smiles is not None
        and len(test_smiles) > 0
    ):
        train_scores, test_scores = get_train_test_scores(
            estimator, buildconfig, train_X, train_y, test_X, test_y
        )
    else:
        train_scores = get_merged_train_score(estimator, buildconfig, train_X, train_y)
        test_scores = None
    return estimator, train_scores, test_scores
