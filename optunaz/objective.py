import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
import sklearn.model_selection
from apischema import serialize, deserialize

from optunaz.config.build_from_opt import suggest_alg_params
from optunaz.config.optconfig import OptimizationConfig, ModelMode
import optunaz.config.buildconfig as build
from optunaz.descriptors import descriptor_from_config, AnyDescriptor
from optunaz.utils.enums import TrialParams

logger = logging.getLogger(__name__)


classification_scores = (
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    # "f1_samples",  # Samplewise metric, not available outside of multilabel classification.
    "f1_weighted",
    "jaccard",
    "jaccard_macro",
    "jaccard_micro",
    # "jaccard_samples",
    "jaccard_weighted",
    # "neg_log_loss",  # Requires predict_proba, which requires expensive calibration for SVC.
    "precision",
    "precision_macro",
    "precision_micro",
    # "precision_samples",
    "precision_weighted",
    "recall",
    "recall_macro",
    "recall_micro",
    # "recall_samples",
    "recall_weighted",
    "roc_auc",
)


regression_scores = (
    "explained_variance",
    "max_error",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    # "neg_mean_squared_log_error",  # Does not work for negative inputs.
    "neg_median_absolute_error",
    "r2",
)


@dataclass
class Objective:
    optconfig: OptimizationConfig
    train_smiles: List[str]
    train_y: np.ndarray

    def __call__(self, trial):

        build_alg = self._get_estimator(trial)
        estimator = build_alg.estimator()

        X = self._get_descriptor(trial)

        mode = self.optconfig.settings.mode
        score_for_objective = f"test_{self.optconfig.settings.scoring}"
        if mode == ModelMode.REGRESSION:
            scoring = regression_scores
        elif mode == ModelMode.CLASSIFICATION:
            scoring = classification_scores
        else:
            raise ValueError(f"Unrecognized mode: {mode}.")

        try:
            scores = sklearn.model_selection.cross_validate(
                estimator=estimator,
                X=X,
                y=self.train_y,
                n_jobs=self.optconfig.settings.n_jobs,
                cv=self.optconfig.settings.cross_validation,
                scoring=scoring,
                return_train_score=True,
            )
        except TypeError as e:
            raise TypeError(
                f"CV failed for alg {build_alg}, estimator {estimator}, {e}"
            ) from e

        # Add attributes to the trial to be accessed later.
        train_scores = {k: scores["train_" + k].tolist() for k in scoring}
        test_scores = {k: scores["test_" + k].tolist() for k in scoring}
        trial.set_user_attr(key="train_scores", value=train_scores)
        trial.set_user_attr(key="test_scores", value=test_scores)

        # Take mean test score for all CV folds and return it as objective.
        return scores[score_for_objective].mean()

    def _get_estimator(self, trial) -> build.AnyAlgorithm:
        alg_choices = [alg.name for alg in self.optconfig.algorithms]
        alg_name = trial.suggest_categorical(
            TrialParams.ALGORITHM_NAME.value, alg_choices
        )

        # Get alg from list by alg's name.
        alg = next(alg for alg in self.optconfig.algorithms if alg.name == alg_name)
        build_alg = suggest_alg_params(trial, alg)

        return build_alg

    def _get_descriptor(self, trial):
        """Calculates a descriptor (fingerprint) for the trial."""

        # Convert descriptor config to `str` to store name+params in `trial`.
        descriptor_choices = [
            json.dumps(serialize(d)) for d in self.optconfig.descriptors
        ]
        descriptor_str = trial.suggest_categorical(
            TrialParams.DESCRIPTOR.value, descriptor_choices
        )

        # Get back the object.
        descriptor_dict = json.loads(descriptor_str)
        descriptor = deserialize(AnyDescriptor, descriptor_dict, additional_properties=True)

        X = descriptor_from_config(self.train_smiles, descriptor)
        return X
