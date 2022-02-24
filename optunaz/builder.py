from optunaz.config.buildconfig import BuildConfig
from optunaz.descriptors import descriptor_from_config
from optunaz.evaluate import get_merged_train_score, get_train_test_scores


def build(buildconfig: BuildConfig, merge_train_and_test_data: bool = False):
    """Build regressor or classifier model and return it."""

    estimator = buildconfig.algorithm.estimator()
    if merge_train_and_test_data:
        train_smiles, train_y = buildconfig.data.get_merged_sets()
    else:
        train_smiles, train_y, _, _ = buildconfig.data.get_sets()

    train_X = descriptor_from_config(train_smiles, buildconfig.descriptor)

    estimator.fit(train_X, train_y)

    if merge_train_and_test_data:
        train_scores = get_merged_train_score(estimator, buildconfig)
        test_scores = None
    else:
        train_scores, test_scores = get_train_test_scores(estimator, buildconfig)

    return estimator, train_scores, test_scores
