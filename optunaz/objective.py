import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import warnings
from apischema import serialize, deserialize
from joblib import Memory, effective_n_jobs
from functools import partial

import sklearn.model_selection
from sklearn.metrics import make_scorer
from sklearn.exceptions import UndefinedMetricWarning

from optunaz.config.build_from_opt import (
    suggest_alg_params,
    suggest_aux_params,
    check_invalid_descriptor_param,
)
from optunaz.config.optconfig import OptimizationConfig, ModelMode
import optunaz.config.buildconfig as build
from optunaz.descriptors import (
    descriptor_from_config,
    AnyDescriptor,
    ScalingFittingError,
)
from optunaz.utils import remove_failed_idx
from optunaz.utils.enums import TrialParams
from optuna import TrialPruned
from optuna.trial import TrialState
from optunaz.metircs import auc_pr_cal, bedroc_score, concordance_index

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("chemprop").disabled = True
logging.getLogger("train").disabled = True

logger = logging.getLogger(__name__)

sklearn.metrics._scorer._SCORERS.update(
    {
        "auc_pr_cal": make_scorer(auc_pr_cal, response_method="predict_proba"),
        "bedroc": make_scorer(bedroc_score, response_method="predict_proba"),
        "concordance_index": make_scorer(
            concordance_index, response_method="predict_proba"
        ),
    }
)

classification_scores = {
    "accuracy": "accuracy",
    "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "f1_weighted": "f1_weighted",
    "jaccard": "jaccard",
    "jaccard_macro": "jaccard_macro",
    "jaccard_micro": "jaccard_micro",
    "jaccard_weighted": "jaccard_weighted",
    "neg_brier_score": "neg_brier_score",
    "precision": "precision",
    "precision_macro": "precision_macro",
    "precision_micro": "precision_micro",
    "precision_weighted": "precision_weighted",
    "recall": "recall",
    "recall_macro": "recall_macro",
    "recall_micro": "recall_micro",
    "recall_weighted": "recall_weighted",
    "roc_auc": "roc_auc",
    "auc_pr_cal": make_scorer(auc_pr_cal, response_method="predict_proba"),
    "bedroc": make_scorer(bedroc_score, response_method="predict_proba"),
    "concordance_index": make_scorer(
        concordance_index, response_method="predict_proba"
    ),
}


regression_scores = (
    "explained_variance",
    "max_error",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_median_absolute_error",
    "r2",
)


class NoValidDescriptors(Exception):
    """Raised when none of the supplied descriptors are compatible with any of the supplied algorithms"""

    pass


def null_scores(scoring):
    null_scoring = {k: [float("nan")] for k in scoring}
    return null_scoring


@dataclass
class Objective:
    optconfig: OptimizationConfig
    train_smiles: List[str]
    train_y: np.ndarray
    train_aux: np.ndarray = None
    cache: Optional[Memory] = None

    def __call__(self, trial):
        # Set up the mode for reg or cls
        mode = self.optconfig.settings.mode
        score_for_objective = f"test_{self.optconfig.settings.scoring}"
        minimise_std_dev_objective = self.optconfig.settings.minimise_std_dev
        if mode == ModelMode.REGRESSION:
            scoring = regression_scores
        elif mode == ModelMode.CLASSIFICATION:
            scoring = classification_scores
        else:
            raise ValueError(f"Unrecognized mode: {mode}.")

        # Ensure train/test scores are set (NaN), since mlflow.py always expects them
        trial.set_user_attr(key="train_scores", value=null_scores(scoring))
        trial.set_user_attr(key="test_scores", value=null_scores(scoring))
        trial.set_user_attr(key="trial_ran", value=False)

        # Get algo & descriptor from Optuna, get valid descriptor combo
        self._validate_algos()
        build_alg = self._get_estimator(trial)
        try:
            estimator = build_alg.estimator()
        except FileNotFoundError as e:
            raise TrialPruned(f"Estimator initiation failed for algorithm: {e}")
        try:
            descriptor, valid_descriptors, aux_weight_pc = self._get_descriptor(
                trial, build_alg
            )
            if self.cache is not None:
                _descriptor_from_config = partial(
                    descriptor_from_config, cache=self.cache
                )
                cache_desc_from_conf = self.cache.cache(_descriptor_from_config)
                X, failed_idx = cache_desc_from_conf(self.train_smiles, descriptor)
            else:
                X, failed_idx = descriptor_from_config(self.train_smiles, descriptor)
            if len(X) == 0:
                raise ValueError
        except (ScalingFittingError, ValueError) as e:
            raise TrialPruned(f"Descriptor generation failed for descriptor: {e}")
        train_y, train_smiles, train_aux = remove_failed_idx(
            failed_idx, self.train_y, self.train_smiles, self.train_aux
        )
        if train_aux is not None:
            X = np.hstack((X, train_aux))
        if len(failed_idx) > 0:
            logger.warning(
                f"Descriptor [{descriptor}] for trial [{trial}] has {len(failed_idx)} \
                erroneous smiles at indices {failed_idx}"
            )
        if len(X) < self.optconfig.settings.cross_validation:
            raise TrialPruned(
                f"Issue with structures or descriptor config. Insufficient descriptors ({len(X)} generated for: "
                f"{descriptor.name}"
            )

        # Check trial duplication, prune if this is detected.
        for t in trial.study.trials:
            if t.state == TrialState.COMPLETE and t.params == trial.params:
                # Set the pruned trial test/train scores to the duplicated trial
                trial.set_user_attr(
                    key="train_scores", value=t.user_attrs["train_scores"]
                )
                trial.set_user_attr(
                    key="test_scores", value=t.user_attrs["test_scores"]
                )
                if hasattr(t, "values"):
                    print(f"Duplicated trial: {trial.params}, return {t.values}")
                else:
                    print(f"Duplicated trial: {trial.params}, return {t.value}")
                # Raising `TrialPruned` instead of just 'return t.value' means that the
                # sampler is more likely to avoid evaluating identical parameters again.
                # See stackoverflow.com/questions/58820574 discussion wrt this issue
                raise TrialPruned("Duplicate parameter set")

        # CV is only attempted when the descriptor is compatible with the algo
        if type(descriptor) in valid_descriptors:
            # Auxiliary weight is applied here, if used, and if algorithm supports this
            if aux_weight_pc is not None:
                if hasattr(estimator, "aux_weight_pc"):
                    estimator.aux_weight_pc = aux_weight_pc
                elif hasattr(estimator, "base_estimator") and hasattr(
                    estimator.base_estimator, "aux_weight_pc"
                ):
                    estimator.base_estimator.aux_weight_pc = aux_weight_pc

            cv = self.optconfig.settings.cv_split_strategy.get_sklearn_splitter(
                n_splits=self.optconfig.settings.cross_validation
            )

            n_jobs = effective_n_jobs(self.optconfig.settings.n_jobs)
            # ensure ChemProp uses parallelisation within trial, not cross_validate
            if (
                hasattr(estimator, "num_workers")
                or hasattr(estimator, "estimator")
                and hasattr(estimator.estimator, "num_workers")
            ):
                n_jobs = 1

            try:
                scores = sklearn.model_selection.cross_validate(
                    estimator=estimator,
                    X=X,
                    y=train_y,
                    n_jobs=n_jobs,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=True,
                )
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"CV failed for alg {build_alg}, estimator {estimator}:  {e}"
                )

            # Add attributes to the trial to be accessed later.
            train_scores = {k: scores["train_" + k].tolist() for k in scoring}
            test_scores = {k: scores["test_" + k].tolist() for k in scoring}
            trial.set_user_attr(key="train_scores", value=train_scores)
            trial.set_user_attr(key="test_scores", value=test_scores)
            trial.set_user_attr(key="trial_ran", value=True)
            # Take mean test score for all CV folds and return it as objective.
            if minimise_std_dev_objective:
                return (
                    scores[score_for_objective].mean(),
                    scores[score_for_objective].std(),
                )
            else:
                return scores[score_for_objective].mean()

        # Otherwise, the descriptor is not compatible, and is handled here
        else:
            # Return the _worst_ possible score, since Optuna does not allow pruning 1st trials.
            # FYI: Returning NaN would result in 'ValueError: No trials are completed yet' due to
            # calling Optuna attribute 'study.best_trial' in build_from_opt.py.
            if len(trial.study.trials) == 1:
                if minimise_std_dev_objective:
                    if trial.study.directions[0].name == "MAXIMIZE":
                        return -np.inf, np.inf
                    else:
                        return np.inf, np.inf
                else:
                    if trial.study.direction.name == "MAXIMIZE":
                        return -np.inf
                    else:
                        return np.inf
            # Otherwise, this trial is not the 1st trial & Optuna allows pruning this trial. Pruning guides
            # the optimiser away from invalid subspaces (incompatible algo/descriptor pairs).
            # See stackoverflow.com/questions/70681612 for a discussion implementing this solution.
            else:
                raise TrialPruned("Incompatible subspace")

    def _validate_algos(self):
        """Ensures algorithms are compatible with the input data before starting objective"""
        # additional validation for prf
        possible_algs = [alg.name for alg in self.optconfig.algorithms]
        cp_regex = re.compile("ChemProp.*?Regressor")
        if (
            "PRFClassifier" in possible_algs
            and not np.logical_and(self.train_y >= 0, self.train_y <= 1).all()
        ):
            raise ValueError(
                "PRFClassifier supplied but response column outside [0.0-1.0] acceptable range. "
                f"Response max: {self.train_y.max()}, response min: {self.train_y.min()} "
            )
        elif any([re.match(cp_regex, alg) for alg in possible_algs]) and set(
            self.train_y
        ) == {0, 1}:
            raise ValueError(
                "ChemProp regressor supplied but response column appears classification."
            )
        return

    def _get_estimator(self, trial) -> build.AnyAlgorithm:
        """Calculates an estimator (algorithm) for the trial."""
        alg_choices = [alg.name for alg in self.optconfig.algorithms]
        alg_name = trial.suggest_categorical(
            TrialParams.ALGORITHM_NAME.value, alg_choices
        )
        # Get alg from list by alg's hash.
        hash_choices = [
            alg.hash for alg in self.optconfig.algorithms if alg.name == alg_name
        ]
        alg_hash = trial.suggest_categorical(
            f"{alg_name}_{TrialParams.ALGORITHM_HASH.value}", hash_choices
        )
        trial.set_user_attr("alg_hash", alg_hash)
        alg = next(alg for alg in self.optconfig.algorithms if alg.hash == alg_hash)
        build_alg = suggest_alg_params(trial, alg)
        return build_alg

    def _get_descriptor(self, trial, algo) -> [AnyDescriptor, tuple, int | None]:
        """Calculates a descriptor (fingerprint) for the trial."""

        valid_descriptors = check_invalid_descriptor_param(algo)

        # Check that there are possible choices first
        possible_choices = [
            d
            for d in self.optconfig.descriptors
            if isinstance(d, Union[valid_descriptors])
        ]

        # Raise value error so the user must provide some possible algo/descriptor combinations
        if len(possible_choices) == 0:
            raise NoValidDescriptors(
                "None of the supplied descriptors: "
                f"{[desc.name for desc in self.optconfig.descriptors]} "
                f"are compatible with the supplied algo: {algo.parameters}."
            )

        # Convert descriptor config to `str` to store name+params in `trial`.
        descriptor_choices = [
            json.dumps(serialize(d)) for d in self.optconfig.descriptors
        ]

        # Ideally we could suggest_categorical from possible_choices here, and negate the workarounds
        # above, but currently CategoricalDistribution suggestor has no 'dynamic value space' support.
        # See https://github.com/optuna/optuna/issues/2328 for discussion on the issue
        descriptor_str = trial.suggest_categorical(
            TrialParams.DESCRIPTOR.value, descriptor_choices
        )

        # Get back the object.
        descriptor_dict = json.loads(descriptor_str)
        descriptor = deserialize(
            AnyDescriptor, descriptor_dict, additional_properties=True
        )
        # Suggest aux params if supported by descriptor
        aux_params = suggest_aux_params(trial, descriptor)

        return descriptor, valid_descriptors, aux_params
