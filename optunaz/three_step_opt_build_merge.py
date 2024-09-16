import logging
from typing import Optional, Union
import math

import json
from apischema import serialize

from joblib import Memory

from optunaz.builder import build
from optunaz.config import OptimizationDirection
from optunaz.config.build_from_opt import buildconfig_from_trial
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import (
    OptimizationConfig,
    ChemPropClassifier,
    ChemPropRegressor,
)
from optunaz.model_writer import save_model
from optunaz.objective import Objective
from optunaz.utils.enums import StudyUserAttrs, TrialParams
from optunaz.utils.tracking import InternalTrackingCallback, track_build

logger = logging.getLogger(__name__)


def split_optimize(optconfig: OptimizationConfig):
    """Split Hyperparameter runs into non-chemprop and chemprop runs for Optuna."""
    import copy
    from optunaz.config.optconfig import (
        AnyChemPropAlgorithm,
        CalibratedClassifierCVWithVA,
    )
    from optunaz.descriptors import SmilesBasedDescriptor

    configs = []

    # populate the optconfig for non-chemprop and chemprop algo's
    for cond in [False, True]:
        cfg = copy.deepcopy(optconfig)
        algos = []
        for algo in cfg.algorithms:
            estimator = type(algo)
            if estimator == CalibratedClassifierCVWithVA:
                estimator = type(getattr(algo.parameters, "estimator"))
            if (estimator in AnyChemPropAlgorithm) == cond:
                algos.append(algo)
        cfg.algorithms = algos
        cfg.descriptors = [
            desc
            for desc in cfg.descriptors
            if (type(desc) in SmilesBasedDescriptor.__args__) == cond
        ]
        if len(cfg.algorithms) != 0 and len(cfg.descriptors) != 0:
            configs.append(cfg)
    return configs


def base_chemprop_params(alg):
    """Used to enqueue an initial ChemProp run that captures sensible defaults as defined by original authors.
    A Check is performed to ensure any parameters outside valid Optuna subspace are popped from fixed parameters.
    """
    from optunaz.algorithms.chem_prop import BaseChemProp
    from optunaz.config.build_from_opt import encode_name
    from functools import partial

    _encode_name = partial(encode_name, hash=alg.hash)
    base_cp = BaseChemProp()
    fixed_params = {
        param: getattr(base_cp, param)
        for param in alg.parameters.__dict__.keys()
        if param not in ["epochs", "ensemble_size"]
    }
    # Remove recommended fixed parameters that would be outside the valid Optuna subspace provided by user optconfig
    for param in list(fixed_params.keys()):
        thisattr = getattr(alg.parameters, param)
        # Recommended values outside user config are dropped here
        if hasattr(thisattr, "low"):
            if not thisattr.low <= fixed_params[param] <= thisattr.high:
                fixed_params.pop(param)
        # Recommended items not within enum of the user config are dropped here
        else:
            if not fixed_params[param] in [attr.value for attr in thisattr]:
                fixed_params.pop(param)
    fixed_params = {
        _encode_name(param): value for param, value in fixed_params.items()
    }  # add algo hash
    return {
        **fixed_params,
        **{
            "algorithm_name": alg.name,
            f"{alg.name}_{TrialParams.ALGORITHM_HASH.value}": alg.hash,
        },
    }


def run_study(
    optconfig: OptimizationConfig,
    study_name,
    objective,
    n_startup_trials,
    n_trials,
    seed,
    storage=True,
    trial_number_offset=0,
):
    """Run an Optuna study"""
    # Import here to not "spill" dependencies into pickled/dilled models.
    import optuna
    from optuna.samplers import TPESampler

    sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)

    if storage:
        storage = optconfig.settings.optuna_storage
        load_if_exists = True
    else:
        storage = None
        load_if_exists = False

    if optconfig.settings.minimise_std_dev:
        study = optuna.create_study(
            storage=storage,
            directions=[
                optconfig.settings.direction,
                OptimizationDirection.MINIMIZATION,
            ],
            study_name=study_name,
            sampler=sampler,
            load_if_exists=load_if_exists,
        )
    else:
        study = optuna.create_study(
            storage=storage,
            direction=optconfig.settings.direction,
            study_name=study_name,
            sampler=sampler,
            load_if_exists=load_if_exists,
        )

    study.set_user_attr(StudyUserAttrs.OPTCONFIG, serialize(optconfig))
    if isinstance(objective.cache, Memory):
        study.set_user_attr("cache", objective.cache.location)
    else:
        study.set_user_attr("cache", objective.cache)
    callbacks = []
    if optconfig.settings.track_to_mlflow:
        from optunaz.utils.mlflow import MLflowCallback

        callbacks.append(
            MLflowCallback(optconfig=optconfig, trial_number_offset=trial_number_offset)
        )
    if optconfig.settings.tracking_rest_endpoint is not None:
        callbacks.append(
            InternalTrackingCallback(
                optconfig=optconfig, trial_number_offset=trial_number_offset
            )
        )

    if n_trials >= 1:
        for alg in optconfig.algorithms:
            if isinstance(alg, Union[ChemPropClassifier, ChemPropRegressor]):
                # Initial ChemProp trials are first directed to sensible defaults, as defined by the original authors
                sensible_default = base_chemprop_params(alg)
                study.enqueue_trial(sensible_default)
                logging.info(
                    f"Enqueued ChemProp manual trial with sensible defaults: {sensible_default}"
                )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
    )

    # NB: A master_study will have 0 trials, otherwise we ensure that any trials ran
    if n_trials != 0:
        if (~study.trials_dataframe()["user_attrs_trial_ran"]).all():
            logging.warning(
                f"None of the trials were able to finish: {study.trials_dataframe()}"
            )
            raise ValueError("Exiting since no trials returned values")
    return study


def optimize(optconfig: OptimizationConfig, study_name: Optional[str] = None):
    """Step 1. Hyperparameter optimization using Optuna."""

    train_smiles, train_y, train_aux, _, _, _ = optconfig.data.get_sets()
    n_startup_trials = optconfig.settings.n_startup_trials
    n_trials = optconfig.settings.n_trials
    n_chemprop_trials = optconfig.settings.n_chemprop_trials
    random_seed = optconfig.settings.random_seed
    objective = Objective(
        optconfig=optconfig,
        train_smiles=train_smiles,
        train_y=train_y,
        train_aux=train_aux,
        cache=optconfig._cache,
    )

    if optconfig.settings.split_chemprop:
        # Separate optuna runs for Chemprop are handled here. The approach is to have two optuna runs, for shallow and
        # chemprop algorithms, respectively. Once complete, they are added to a master study to avoid dynamic subspace
        # checks. Each study is able to callback trial results with the use of an offset
        master_study = run_study(optconfig, study_name, objective, 0, 0, random_seed)
        try:
            trial_number_offset = 0
            algo_dist = tuple(serialize(i.name) for i in optconfig.algorithms)
            descript_dist = tuple(
                json.dumps(serialize(d)) for d in optconfig.descriptors
            )
            # enumerate through the shallow and chemprop studies, respectively
            for cfg_idx, cfg in enumerate(split_optimize(optconfig)):
                sub_objective = Objective(
                    optconfig=cfg,
                    train_smiles=train_smiles,
                    train_y=train_y,
                    train_aux=train_aux,
                    cache=optconfig._cache,
                )
                study = run_study(
                    cfg,
                    f"study_name_{cfg_idx}",
                    sub_objective,
                    n_startup_trials,
                    n_trials,
                    random_seed,
                    storage=False,
                    trial_number_offset=trial_number_offset,
                )
                # manually set the distributions to avoid dynamic subspace error
                for st_idx, st in enumerate(study.get_trials(deepcopy=False)):
                    try:
                        st.distributions["descriptor"].choices = descript_dist
                        st.distributions["algorithm_name"].choices = algo_dist
                        study.trials[st_idx] = st
                    except KeyError:
                        pass  # skip trials that did not get a descriptor or algorithm choice
                # set parameters for next chemprop study (currently share n chemprop trials with
                if cfg_idx == 0:
                    n_chemprop_shared_trials = n_chemprop_trials / 2
                    studies = study
                    trial_number_offset = len(study.get_trials())
                    n_startup_trials = math.floor(n_chemprop_shared_trials)
                    n_trials = math.ceil(n_chemprop_shared_trials)
                # add the chemprop results to the existing study
                else:
                    studies.add_trials(study.trials)
            # update the master study with all trials
            for st_idx, st in enumerate(studies.get_trials(deepcopy=False)):
                master_study.add_trial(st)
            return master_study
        except UnboundLocalError:
            raise UnboundLocalError("No valid subspaces were found, check your config")
    else:
        return run_study(
            optconfig, study_name, objective, n_startup_trials, n_trials, random_seed
        )


def buildconfig_best(study):
    try:
        return buildconfig_from_trial(study, study.best_trial)
    except RuntimeError:
        return buildconfig_from_trial(study, study.best_trials[0])


def log_scores(scores, main_score, label: str):
    main_score_val = scores.get(main_score, None)
    if main_score_val is not None:
        logger.info(f"{label.capitalize()} score {main_score}: {main_score_val}")
    logger.info(
        f"All {label} scores: { {k: round(number=v, ndigits=3) for k, v in scores.items()} }"
    )


def build_best(
    buildconfig: BuildConfig,
    outfname,
    cache: Optional[Memory] = None,
):
    """Step 2. Build. Train a model with the best hyperparameters."""

    model, train_scores, test_scores = build(buildconfig, cache=cache)
    qsartuna_model = save_model(
        model,
        buildconfig,
        outfname,
        train_scores,
        test_scores,
    )

    # print model characteristics
    logger.info(f"Model: {outfname}")
    log_scores(train_scores, buildconfig.settings.scoring, "train")
    if test_scores is not None:
        log_scores(test_scores, buildconfig.settings.scoring, "test")

    if buildconfig.settings.tracking_rest_endpoint is not None:
        track_build(qsartuna_model, buildconfig, test_scores)

    return buildconfig


def build_merged(
    buildconfig: BuildConfig,
    outfname,
    cache: Optional[Memory] = None,
):
    """Step 3. Merge datasets and re-train the model."""

    model, train_scores, test_scores = build(
        buildconfig, merge_train_and_test_data=True, cache=cache
    )
    save_model(
        model,
        buildconfig,
        outfname,
        train_scores,
        test_scores,
    )

    # Print model characteristics.
    logger.info(f"Model: {outfname}")
    log_scores(train_scores, buildconfig.settings.scoring, "train")
