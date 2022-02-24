import json

from apischema import deserialize
from optuna import Study
from optuna.trial import FrozenTrial

from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.descriptors import MolDescriptor
from optunaz.utils import mkdict
from optunaz.utils.enums import StudyUserAttrs, TrialParams

import optunaz.config.optconfig as opt
import optunaz.config.buildconfig as build
from optunaz.utils.enums.configuration_enum import ConfigurationEnum


_CE = ConfigurationEnum()


def buildconfig_from_trial(study: Study, trial: FrozenTrial) -> BuildConfig:

    optconfig_json = study.user_attrs.get(StudyUserAttrs.OPTCONFIG, None)
    if optconfig_json is None:
        raise ValueError(
            "Study does not have a user attribute with Optimization Configuration."
        )
    optconfig = deserialize(OptimizationConfig, optconfig_json)

    descriptor_json = trial.params[TrialParams.DESCRIPTOR]
    descriptor_dict = json.loads(descriptor_json)
    descriptor = deserialize(MolDescriptor, descriptor_dict)

    algorithm_dict = {
        "name": trial.params[TrialParams.ALGORITHM_NAME],
        "parameters": mkdict(trial.params),
    }
    algorithm = deserialize(
        build.AnyAlgorithm, algorithm_dict, additional_properties=True
    )

    return BuildConfig(
        data=optconfig.data,
        descriptor=descriptor,
        algorithm=algorithm,
        metadata=BuildConfig.Metadata(
            cross_validation=optconfig.settings.cross_validation,
            shuffle=optconfig.settings.shuffle,
            best_trial=study.best_trial.number,
            best_value=study.best_value,
            n_trials=optconfig.settings.n_trials,
        ),
        settings=BuildConfig.Settings(
            mode=optconfig.settings.mode,
            scoring=optconfig.settings.scoring,
            direction=optconfig.settings.direction,
            n_trials=optconfig.settings.n_trials,
        ),
    )


def suggest_alg_params(trial: FrozenTrial, alg: opt.AnyAlgorithm) -> build.AnyAlgorithm:
    para = alg.parameters

    if isinstance(alg, opt.AdaBoostClassifier):
        n_estimators = trial.suggest_int(
            name=_CE.ALGORITHMS_ADABOOSTCLASSIFIER_N_ESTIMATORS,
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        learning_rate = trial.suggest_uniform(
            name=_CE.ALGORITHMS_ADABOOSTCLASSIFIER_LEARNING_RATE,
            low=para.learning_rate.low,
            high=para.learning_rate.high,
        )
        return build.AdaBoostClassifier.new(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
    elif isinstance(alg, opt.Lasso):
        alpha = trial.suggest_uniform(
            name=_CE.ALGORITHMS_LASSO_ALPHA,
            low=para.alpha.low,
            high=para.alpha.high,
        )
        return build.Lasso.new(alpha=alpha)
    elif isinstance(alg, opt.LogisticRegression):
        solver = trial.suggest_categorical(
            name=_CE.ALGORITHMS_LOGISTICREGRESSION_SOLVER,
            choices=para.solver,
        )
        lg_c = trial.suggest_loguniform(
            name=_CE.ALGORITHMS_LOGISTICREGRESSION_C,
            low=para.C.low,
            high=para.C.high,
        )
        return build.LogisticRegression.new(solver=solver, C=lg_c)
    elif isinstance(alg, opt.PLSRegression):
        n_components = trial.suggest_int(
            name=_CE.ALGORITHMS_PLS_N_COMPONENTS,
            low=para.n_components.low,
            high=para.n_components.high,
        )
        return build.PLSRegression.new(n_components=n_components)
    elif isinstance(alg, opt.RandomForestClassifier):
        max_depth = trial.suggest_int(
            name=_CE.ALGORITHMS_RF_MAX_DEPTH,
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_CE.ALGORITHMS_RF_N_ESTIMATORS,
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        max_features = trial.suggest_categorical(
            name=_CE.ALGORITHMS_RF_MAX_FEATURES, choices=para.max_features
        )
        return build.RandomForestClassifier.new(
            max_depth=max_depth, n_estimators=n_estimators, max_features=max_features
        )
    elif isinstance(alg, opt.RandomForestRegressor):
        max_depth = trial.suggest_int(
            name=_CE.ALGORITHMS_RF_MAX_DEPTH,
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_CE.ALGORITHMS_RF_N_ESTIMATORS,
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        max_features = trial.suggest_categorical(
            name=_CE.ALGORITHMS_RF_MAX_FEATURES, choices=para.max_features
        )
        return build.RandomForestRegressor.new(
            max_depth=max_depth, n_estimators=n_estimators, max_features=max_features
        )
    elif isinstance(alg, opt.Ridge):
        alpha = trial.suggest_uniform(
            name=_CE.ALGORITHMS_RIDGE_ALPHA,
            low=para.alpha.low,
            high=para.alpha.high,
        )
        return build.Ridge.new(alpha=alpha)
    elif isinstance(alg, opt.SVC):
        gamma = trial.suggest_loguniform(
            name=_CE.ALGORITHMS_SVC_GAMMA,
            low=para.gamma.low,
            high=para.gamma.high,
        )
        svc_c = trial.suggest_loguniform(
            name=_CE.ALGORITHMS_SVC_C,
            low=para.C.low,
            high=para.C.high,
        )
        return build.SVC.new(gamma=gamma, C=svc_c)
    elif isinstance(alg, opt.SVR):
        gamma = trial.suggest_loguniform(
            name=_CE.ALGORITHMS_SVR_GAMMA,
            low=para.gamma.low,
            high=para.gamma.high,
        )
        svr_c = trial.suggest_loguniform(
            name=_CE.ALGORITHMS_SVR_C,
            low=para.C.low,
            high=para.C.high,
        )
        return build.SVR.new(C=svr_c, gamma=gamma)
    elif isinstance(alg, opt.XGBRegressor):
        max_depth = trial.suggest_int(
            name=_CE.ALGORITHMS_XGBREGRESSOR_MAX_DEPTH,
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_CE.ALGORITHMS_XGBREGRESSOR_N_ESTIMATORS,
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        learning_rate = trial.suggest_uniform(
            name=_CE.ALGORITHMS_XGBREGRESSOR_LEARNING_RATE,
            low=para.learning_rate.low,
            high=para.learning_rate.high,
        )
        return build.XGBRegressor.new(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Unrecognized algorithm: {alg.__class__}")
