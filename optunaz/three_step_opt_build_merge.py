import logging
from typing import Optional

from apischema import serialize

from optunaz.builder import build
from optunaz.config.build_from_opt import buildconfig_from_trial
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.model_writer import save_model, ModelPersistenceMode
from optunaz.objective import Objective
from optunaz.utils.enums import StudyUserAttrs
from optunaz.utils.mlflow import MLflowCallback
from optunaz.utils.tracking import InternalTrackingCallback, track_build

logger = logging.getLogger(__name__)


def optimize(optconfig: OptimizationConfig, study_name: Optional[str] = None):
    """Step 1. Hyperparameter optimization using Optuna."""

    train_smiles, train_y, _, _ = optconfig.data.get_sets()

    objective = Objective(
        optconfig=optconfig, train_smiles=train_smiles, train_y=train_y
    )

    # Import here to not "spill" dependencies into pickled/dilled models.
    import optuna
    from optuna.samplers import TPESampler

    seed = optconfig.settings.random_seed
    n_startup_trials = optconfig.settings.n_startup_trials
    sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)

    study = optuna.create_study(
        storage=optconfig.settings.optuna_storage,
        direction=optconfig.settings.direction,
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True,
    )
    study.set_user_attr(StudyUserAttrs.OPTCONFIG, serialize(optconfig))
    callbacks = []
    if optconfig.settings.track_to_mlflow:
        callbacks.append(MLflowCallback(optconfig=optconfig))
    if optconfig.settings.tracking_rest_endpoint is not None:
        callbacks.append(InternalTrackingCallback(optconfig=optconfig))
    study.optimize(
        objective,
        n_trials=optconfig.settings.n_trials,
        callbacks=callbacks,
    )
    return study


def buildconfig_best(study):
    return buildconfig_from_trial(study, study.best_trial)


def report_scores(scores, main_score, label: str):
    main_score_val = scores.get(main_score, None)
    if main_score_val is not None:
        logger.info(f"{label.capitalize()} score {main_score}: {main_score_val}")
    logger.info(
        f"All {label} cores: { {k: round(number=v, ndigits=3) for k, v in scores.items()} }"
    )


def build_best(
    buildconfig: BuildConfig, outfname, persist_as: ModelPersistenceMode = None
):
    """Step 2. Build. Train a model with the best hyperparameters."""

    model, train_scores, test_scores = build(buildconfig)
    save_model(
        model, buildconfig.descriptor, buildconfig.settings.mode, outfname, persist_as
    )

    # print model characteristics
    logger.info(f"Model: {outfname}")
    report_scores(train_scores, buildconfig.settings.scoring, "train")
    if test_scores is not None:
        report_scores(test_scores, buildconfig.settings.scoring, "test")

    if buildconfig.settings.tracking_rest_endpoint is not None:
        track_build(model, buildconfig)

    return buildconfig


def build_merged(buildconfig, outfname, persist_as: ModelPersistenceMode = None):
    """Step 3. Merge datasets and re-train the model."""

    model, train_scores, test_scores = build(
        buildconfig, merge_train_and_test_data=True
    )
    save_model(
        model, buildconfig.descriptor, buildconfig.settings.mode, outfname, persist_as
    )

    # Print model characteristics.
    logger.info(f"Model: {outfname}")
    report_scores(train_scores, buildconfig.settings.scoring, "train")
