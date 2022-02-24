from typing import Dict, List

from sklearn.metrics import get_scorer

from optunaz import objective
from optunaz.config import ModelMode
from optunaz.config.buildconfig import BuildConfig
from optunaz.descriptors import descriptor_from_config


def score_all(scores: List[str], estimator, X, y) -> Dict[str, float]:
    result = {s: get_scorer(s)(estimator, X, y) for s in scores}
    return result


def get_scores(mode: ModelMode) -> List[str]:
    if mode == ModelMode.REGRESSION:
        scores = objective.regression_scores
    elif mode == ModelMode.CLASSIFICATION:
        scores = objective.classification_scores
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    return scores


def score_all_smiles(scores, estimator, smiles, descriptor, y):
    X = descriptor_from_config(smiles, descriptor)
    return score_all(scores, estimator, X, y)


def get_train_test_scores(estimator, buildconfig: BuildConfig):
    scores = get_scores(buildconfig.settings.mode)

    train_smiles, train_y, test_smiles, test_y = buildconfig.data.get_sets()

    train_scores = score_all_smiles(
        scores, estimator, train_smiles, buildconfig.descriptor, train_y
    )
    if test_smiles is not None and len(test_smiles) > 0:
        test_scores = score_all_smiles(
            scores, estimator, test_smiles, buildconfig.descriptor, test_y
        )
    else:
        test_scores = None

    return train_scores, test_scores


def get_merged_train_score(estimator, buildconfig: BuildConfig):
    scores = get_scores(buildconfig.settings.mode)

    train_smiles, train_y = buildconfig.data.get_merged_sets()

    train_scores = score_all_smiles(
        scores, estimator, train_smiles, buildconfig.descriptor, train_y
    )

    return train_scores
