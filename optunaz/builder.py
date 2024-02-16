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
    else:
        train_smiles, train_y, train_aux, _, _, _ = buildconfig.data.get_sets()

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

    if merge_train_and_test_data:
        train_scores = get_merged_train_score(estimator, buildconfig, cache=cache)
        test_scores = None
    else:
        train_scores, test_scores = get_train_test_scores(
            estimator, buildconfig, cache=cache
        )

    return estimator, train_scores, test_scores
