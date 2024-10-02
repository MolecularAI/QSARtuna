import json
from joblib import effective_n_jobs
from typing import Union
from functools import partial

from apischema import deserialize, serialize
from optuna import Study
from optuna.trial import FrozenTrial

from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.descriptors import MolDescriptor
from joblib import Memory
from optunaz.utils import mkdict
from optunaz.utils.enums import StudyUserAttrs, TrialParams

import optunaz.config.optconfig as opt
import optunaz.config.buildconfig as build
import optunaz.descriptors as descriptors
from optunaz.utils.enums.configuration_enum import ConfigurationEnum

_CE = ConfigurationEnum()


def set_build_cache(study: Study, optconfig: OptimizationConfig) -> Memory | None:
    """Set the cache to preexisting one from Optimisation, when the number of cores supports this"""
    if effective_n_jobs(optconfig.settings.n_jobs) > 1 and "cache" in study.user_attrs:
        return Memory(study.user_attrs["cache"], verbose=0)
    else:
        return None


def remove_algo_hash(trial: FrozenTrial) -> FrozenTrial:
    """Remove the hash from an Optuna algo param set"""
    trial.params = {
        param_name.split("__")[0]: param_value
        for param_name, param_value in trial.params.items()
    }
    return trial


def buildconfig_from_trial(study: Study, trial: FrozenTrial) -> BuildConfig:
    optconfig_json = study.user_attrs.get(StudyUserAttrs.OPTCONFIG, None)
    if optconfig_json is None:
        raise ValueError(
            "Study does not have a user attribute with Optimization Configuration."
        )
    optconfig = deserialize(OptimizationConfig, optconfig_json)

    trial = remove_algo_hash(trial)
    descriptor_json = trial.params[TrialParams.DESCRIPTOR]
    descriptor_dict = json.loads(descriptor_json)
    descriptor = deserialize(MolDescriptor, descriptor_dict)

    # Aux weight for side information prepared
    aux_weight_pc = trial.params.get(_CE.DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC, 100)

    # Base estimator for calibrated methods are prepared here
    base_estimator = trial.user_attrs.get(
        _CE.ALGORITHMS_CALIBRATEDCLASSIFIERCV_ESTIMATOR
    )
    # Pretrained model for pretrained ChemProp methods are prepared here
    pretrained_model = trial.user_attrs.get(
        _CE.ALGORITHMS_CHEMPROP_PRETRAINED_MODEL, {}
    )
    # Parameter dictionary for calibrated CV methods are prepared here
    calibrated_params = trial.user_attrs.get(
        _CE.ALGORITHMS_CALIBRATEDCLASSIFIERCV_PARAMS, {}
    )
    if base_estimator:
        base_estimator[_CE.GENERAL_PARAMETERS][
            _CE.DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC
        ] = aux_weight_pc

    algorithm_dict = {
        _CE.GENERAL_NAME: trial.params.get(_CE.GENERAL_ALGORITHM_NAME),
        _CE.GENERAL_PARAMETERS: mkdict(
            {
                **trial.params,
                **calibrated_params,
                **{
                    _CE.ALGORITHMS_ESTIMATOR: base_estimator,
                    _CE.DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC: aux_weight_pc,
                    _CE.ALGORITHMS_CHEMPROP_PRETRAINED_MODEL: pretrained_model,
                },
            }
        ),
    }

    algorithm = deserialize(
        build.AnyAlgorithm, algorithm_dict, additional_properties=True
    )
    if optconfig.settings.minimise_std_dev:
        best_trial = study.best_trials[0].number
        best_value = study.best_trials[0].values[0]
    else:
        best_trial = study.best_trial.number
        best_value = study.best_value
    return BuildConfig(
        data=optconfig.data,
        descriptor=descriptor,
        algorithm=algorithm,
        metadata=BuildConfig.Metadata(
            name=optconfig.name,
            cross_validation=optconfig.settings.cross_validation,
            shuffle=optconfig.settings.shuffle,
            best_trial=best_trial,
            best_value=best_value,
            n_trials=optconfig.settings.n_trials,
        ),
        settings=BuildConfig.Settings(
            mode=optconfig.settings.mode,
            scoring=optconfig.settings.scoring,
            direction=optconfig.settings.direction,
            n_trials=optconfig.settings.n_trials,
        ),
    )


def encode_name(CEname, hash=hash):
    """Encode the parameter names with a hash to enable multi-parameter optimisation"""
    return f"{CEname}__{hash}"


def suggest_alg_params(trial: FrozenTrial, alg: opt.AnyAlgorithm) -> build.AnyAlgorithm:
    para = alg.parameters
    _encode_name = partial(encode_name, hash=alg.hash)

    if isinstance(alg, opt.AdaBoostClassifier):
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_ADABOOSTCLASSIFIER_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        learning_rate = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_ADABOOSTCLASSIFIER_LEARNING_RATE),
            low=para.learning_rate.low,
            high=para.learning_rate.high,
        )
        return build.AdaBoostClassifier.new(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
    elif isinstance(alg, opt.Lasso):
        alpha = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_LASSO_ALPHA),
            low=para.alpha.low,
            high=para.alpha.high,
        )
        return build.Lasso.new(alpha=alpha)
    elif isinstance(alg, opt.KNeighborsClassifier):
        metric = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_KNEIGHBORS_METRIC),
            choices=para.metric,
        )
        n_neighbors = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_KNEIGHBORS_N_NEIGHBORS),
            low=para.n_neighbors.low,
            high=para.n_neighbors.high,
        )
        weights = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_KNEIGHBORS_WEIGHTS),
            choices=para.weights,
        )
        return build.KNeighborsClassifier.new(
            metric=metric, n_neighbors=n_neighbors, weights=weights
        )
    elif isinstance(alg, opt.KNeighborsRegressor):
        metric = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_KNEIGHBORS_METRIC),
            choices=para.metric,
        )
        n_neighbors = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_KNEIGHBORS_N_NEIGHBORS),
            low=para.n_neighbors.low,
            high=para.n_neighbors.high,
        )
        weights = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_KNEIGHBORS_WEIGHTS),
            choices=para.weights,
        )
        return build.KNeighborsRegressor.new(
            metric=metric, n_neighbors=n_neighbors, weights=weights
        )
    elif isinstance(alg, opt.LogisticRegression):
        solver = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_LOGISTICREGRESSION_SOLVER),
            choices=para.solver,
        )
        lg_c = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_LOGISTICREGRESSION_C),
            low=para.C.low,
            high=para.C.high,
            log=True,
        )
        return build.LogisticRegression.new(solver=solver, C=lg_c)
    elif isinstance(alg, opt.PLSRegression):
        n_components = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_PLSREGRESSION_N_COMPONENTS),
            low=para.n_components.low,
            high=para.n_components.high,
        )
        return build.PLSRegression.new(n_components=n_components)
    elif isinstance(alg, opt.RandomForestClassifier):
        max_depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_RF_MAX_DEPTH),
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_RF_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        max_features = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_RF_MAX_FEATURES),
            choices=para.max_features,
        )
        return build.RandomForestClassifier.new(
            max_depth=max_depth, n_estimators=n_estimators, max_features=max_features
        )
    elif isinstance(alg, opt.RandomForestRegressor):
        max_depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_RF_MAX_DEPTH),
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_RF_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        max_features = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_RF_MAX_FEATURES),
            choices=para.max_features,
        )
        return build.RandomForestRegressor.new(
            max_depth=max_depth, n_estimators=n_estimators, max_features=max_features
        )
    elif isinstance(alg, opt.Ridge):
        alpha = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_RIDGE_ALPHA),
            low=para.alpha.low,
            high=para.alpha.high,
        )
        return build.Ridge.new(alpha=alpha)
    elif isinstance(alg, opt.SVC):
        gamma = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_SVC_GAMMA),
            low=para.gamma.low,
            high=para.gamma.high,
            log=True,
        )
        svc_c = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_SVC_C),
            low=para.C.low,
            high=para.C.high,
            log=True,
        )
        return build.SVC.new(gamma=gamma, C=svc_c)
    elif isinstance(alg, opt.SVR):
        gamma = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_SVR_GAMMA),
            low=para.gamma.low,
            high=para.gamma.high,
            log=True,
        )
        svr_c = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_SVR_C),
            low=para.C.low,
            high=para.C.high,
            log=True,
        )
        return build.SVR.new(C=svr_c, gamma=gamma)
    elif isinstance(alg, opt.XGBRegressor):
        max_depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_XGBREGRESSOR_MAX_DEPTH),
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_XGBREGRESSOR_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        learning_rate = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBREGRESSOR_LEARNING_RATE),
            low=para.learning_rate.low,
            high=para.learning_rate.high,
        )
        return build.XGBRegressor.new(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
    elif isinstance(alg, opt.PRFClassifier):
        max_depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_PRF_MAX_DEPTH),
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_PRF_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        max_features = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_PRF_MAX_FEATURES),
            choices=para.max_features,
        )
        min_py_sum_leaf = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_PRF_MINPYSUMLEAF),
            low=para.min_py_sum_leaf.low,
            high=para.min_py_sum_leaf.high,
        )
        use_py_gini = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_PRF_USE_PY_GINI),
            low=para.use_py_gini,
            high=para.use_py_gini,
        )
        use_py_leafs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_PRF_USE_PY_LEAFS),
            low=para.use_py_leafs,
            high=para.use_py_leafs,
        )
        return build.PRFClassifier.new(
            max_depth=max_depth,
            n_estimators=n_estimators,
            max_features=max_features,
            min_py_sum_leaf=min_py_sum_leaf,
            use_py_gini=use_py_gini,
            use_py_leafs=use_py_leafs,
        )
    elif isinstance(alg, opt.ChemPropRegressor):
        activation = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_ACTIVATION),
            choices=para.activation,
        )
        aggregation = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_AGGREGATION),
            choices=para.aggregation,
        )
        aggregation_norm = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_AGGREGATION_NORM),
            low=para.aggregation_norm.low,
            high=para.aggregation_norm.high,
            step=para.aggregation_norm.q,
        )
        batch_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_BATCH_SIZE),
            low=para.batch_size.low,
            high=para.batch_size.high,
            step=para.batch_size.q,
        )
        depth = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DEPTH),
            low=para.depth.low,
            high=para.depth.high,
            step=para.depth.q,
        )
        dropout = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DROPOUT),
            low=para.dropout.low,
            high=para.dropout.high,
            step=para.dropout.q,
        )
        ensemble_size = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_ENSEMBLE_SIZE),
            low=para.ensemble_size,
            high=para.ensemble_size,
        )
        epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_EPOCHS),
            low=para.epochs,
            high=para.epochs,
        )
        features_generator = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FEATURES_GENERATOR),
            choices=para.features_generator,
        )
        ffn_hidden_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_HIDDEN_SIZE),
            low=para.ffn_hidden_size.low,
            high=para.ffn_hidden_size.high,
            step=para.ffn_hidden_size.q,
        )
        ffn_num_layers = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_NUM_LAYERS),
            low=para.ffn_num_layers.low,
            high=para.ffn_num_layers.high,
            step=para.ffn_num_layers.q,
        )
        final_lr_ratio_exp = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FINAL_LR_RATIO_EXP),
            low=para.final_lr_ratio_exp.low,
            high=para.final_lr_ratio_exp.high,
        )
        hidden_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_HIDDEN_SIZE),
            low=para.hidden_size.low,
            high=para.hidden_size.high,
            step=para.hidden_size.q,
        )
        init_lr_ratio_exp = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_INIT_LR_RATIO_EXP),
            low=para.init_lr_ratio_exp.low,
            high=para.init_lr_ratio_exp.high,
        )
        max_lr_exp = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MAX_LR_EXP),
            low=para.max_lr_exp.low,
            high=para.max_lr_exp.high,
        )
        warmup_epochs_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_WARMUP_EPOCHS_RATIO),
            low=para.warmup_epochs_ratio.low,
            high=para.warmup_epochs_ratio.high,
            step=para.warmup_epochs_ratio.q,
        )
        return build.ChemPropRegressor.new(
            activation=activation,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            batch_size=batch_size,
            depth=depth,
            dropout=dropout,
            ensemble_size=ensemble_size,
            epochs=epochs,
            features_generator=features_generator,
            ffn_hidden_size=ffn_hidden_size,
            ffn_num_layers=ffn_num_layers,
            final_lr_ratio_exp=final_lr_ratio_exp,
            hidden_size=hidden_size,
            init_lr_ratio_exp=init_lr_ratio_exp,
            max_lr_exp=max_lr_exp,
            warmup_epochs_ratio=warmup_epochs_ratio,
        )
    elif isinstance(alg, opt.ChemPropClassifier):
        activation = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_ACTIVATION),
            choices=para.activation,
        )
        aggregation = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_AGGREGATION),
            choices=para.aggregation,
        )
        aggregation_norm = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_AGGREGATION_NORM),
            low=para.aggregation_norm.low,
            high=para.aggregation_norm.high,
            step=para.aggregation_norm.q,
        )
        batch_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_BATCH_SIZE),
            low=para.batch_size.low,
            high=para.batch_size.high,
            step=para.batch_size.q,
        )
        depth = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DEPTH),
            low=para.depth.low,
            high=para.depth.high,
            step=para.depth.q,
        )
        dropout = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DROPOUT),
            low=para.dropout.low,
            high=para.dropout.high,
            step=para.dropout.q,
        )
        ensemble_size = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_ENSEMBLE_SIZE),
            low=para.ensemble_size,
            high=para.ensemble_size,
        )
        epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_EPOCHS),
            low=para.epochs,
            high=para.epochs,
        )
        features_generator = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FEATURES_GENERATOR),
            choices=para.features_generator,
        )
        ffn_hidden_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_HIDDEN_SIZE),
            low=para.ffn_hidden_size.low,
            high=para.ffn_hidden_size.high,
            step=para.ffn_hidden_size.q,
        )
        ffn_num_layers = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_NUM_LAYERS),
            low=para.ffn_num_layers.low,
            high=para.ffn_num_layers.high,
            step=para.ffn_num_layers.q,
        )
        final_lr_ratio_exp = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FINAL_LR_RATIO_EXP),
            low=para.final_lr_ratio_exp.low,
            high=para.final_lr_ratio_exp.high,
        )
        hidden_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_HIDDEN_SIZE),
            low=para.hidden_size.low,
            high=para.hidden_size.high,
            step=para.hidden_size.q,
        )
        init_lr_ratio_exp = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_INIT_LR_RATIO_EXP),
            low=para.init_lr_ratio_exp.low,
            high=para.init_lr_ratio_exp.high,
        )
        max_lr_exp = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MAX_LR_EXP),
            low=para.max_lr_exp.low,
            high=para.max_lr_exp.high,
        )
        warmup_epochs_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_WARMUP_EPOCHS_RATIO),
            low=para.warmup_epochs_ratio.low,
            high=para.warmup_epochs_ratio.high,
            step=para.warmup_epochs_ratio.q,
        )
        return build.ChemPropClassifier.new(
            activation=activation,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            batch_size=batch_size,
            depth=depth,
            dropout=dropout,
            ensemble_size=ensemble_size,
            epochs=epochs,
            features_generator=features_generator,
            ffn_hidden_size=ffn_hidden_size,
            ffn_num_layers=ffn_num_layers,
            final_lr_ratio_exp=final_lr_ratio_exp,
            hidden_size=hidden_size,
            init_lr_ratio_exp=init_lr_ratio_exp,
            max_lr_exp=max_lr_exp,
            warmup_epochs_ratio=warmup_epochs_ratio,
        )
    elif isinstance(alg, opt.ChemPropHyperoptRegressor):
        ensemble_size = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_ENSEMBLE_SIZE),
            low=para.ensemble_size,
            high=para.ensemble_size,
        )
        epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_EPOCHS),
            low=para.epochs,
            high=para.epochs,
        )
        features_generator = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FEATURES_GENERATOR),
            choices=para.features_generator,
        )
        num_iters = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_NUM_ITERS),
            low=para.num_iters,
            high=para.num_iters,
        )
        search_parameter_level = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_SEARCH_PARAMETER_LEVEL),
            choices=para.search_parameter_level,
        )
        return build.ChemPropHyperoptRegressor.new(
            ensemble_size=ensemble_size,
            epochs=epochs,
            features_generator=features_generator,
            num_iters=num_iters,
            search_parameter_level=search_parameter_level,
        )
    elif isinstance(alg, opt.ChemPropHyperoptClassifier):
        ensemble_size = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_ENSEMBLE_SIZE),
            low=para.ensemble_size,
            high=para.ensemble_size,
        )
        epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_EPOCHS),
            low=para.epochs,
            high=para.epochs,
        )
        features_generator = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FEATURES_GENERATOR),
            choices=para.features_generator,
        )
        num_iters = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_NUM_ITERS),
            low=para.num_iters,
            high=para.num_iters,
        )
        search_parameter_level = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_SEARCH_PARAMETER_LEVEL),
            choices=para.search_parameter_level,
        )
        return build.ChemPropHyperoptClassifier.new(
            ensemble_size=ensemble_size,
            epochs=epochs,
            features_generator=features_generator,
            num_iters=num_iters,
            search_parameter_level=search_parameter_level,
        )
    elif isinstance(alg, opt.ChemPropRegressorPretrained):
        frzn = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FRZN), choices=para.frzn
        )
        epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_EPOCHS),
            low=para.epochs.low,
            high=para.epochs.high,
        )
        trial.set_user_attr(
            key=_CE.ALGORITHMS_CHEMPROP_PRETRAINED_MODEL, value=para.pretrained_model
        )

        return build.ChemPropRegressorPretrained.new(
            epochs=epochs,
            frzn=frzn,
            pretrained_model=para.pretrained_model,
        )
    elif isinstance(alg, opt.CalibratedClassifierCVWithVA):
        n_folds = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CALIBRATEDCLASSIFIERCV_N_FOLDS),
            low=para.n_folds,
            high=para.n_folds,
        )
        estimator = suggest_alg_params(trial, para.estimator)
        trial.set_user_attr(
            key=_CE.ALGORITHMS_CALIBRATEDCLASSIFIERCV_ESTIMATOR,
            value=serialize(estimator),
        )
        calibrated_params = {
            _CE.ALGORITHMS_CALIBRATEDCLASSIFIERCV_ENSEMBLE: para.ensemble,
            _CE.ALGORITHMS_CALIBRATEDCLASSIFIERCV_METHOD: para.method,
        }
        trial.set_user_attr(
            key=_CE.ALGORITHMS_CALIBRATEDCLASSIFIERCV_PARAMS, value=calibrated_params
        )

        return build.CalibratedClassifierCVWithVA.new(
            ensemble=para.ensemble,
            estimator=estimator,
            method=para.method,
            n_folds=n_folds,
        )
    elif isinstance(alg, opt.Mapie):
        mapie_alpha = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_ALPHA),
            low=para.mapie_alpha,
            high=para.mapie_alpha,
        )
        estimator = suggest_alg_params(trial, para.estimator)
        trial.set_user_attr(
            key=_CE.ALGORITHMS_MAPIE_ESTIMATOR, value=serialize(estimator)
        )

        return build.Mapie.new(
            estimator=estimator,
            mapie_alpha=mapie_alpha,
        )
    elif isinstance(alg, opt.CustomRegressionModel):
        trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_FILE),
            choices=[para.preexisting_model],
        )
        trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_REFIT_MODEL),
            low=para.refit_model,
            high=para.refit_model,
        )
        return build.CustomRegressionModel.new(
            preexisting_model=para.preexisting_model, refit_model=para.refit_model
        )
    elif isinstance(alg, opt.CustomClassificationModel):
        trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_FILE),
            choices=[para.preexisting_model],
        )
        trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_REFIT_MODEL),
            low=para.refit_model,
            high=para.refit_model,
        )
        return build.CustomClassificationModel.new(
            preexisting_model=para.preexisting_model, refit_model=para.refit_model
        )
    else:
        raise ValueError(f"Unrecognized algorithm: {alg.__class__}")


def suggest_aux_params(trial: FrozenTrial, desc: descriptors.AnyDescriptor):
    para = desc.parameters
    _encode_name = partial(encode_name, hash=trial.user_attrs["alg_hash"])
    # SmilesAndSideInfoFromFile is the only descriptor currently supporting aux params
    if isinstance(desc, descriptors.SmilesAndSideInfoFromFile):
        return trial.suggest_int(
            name=_encode_name(_CE.DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC),
            low=para.aux_weight_pc.low,
            high=para.aux_weight_pc.high,
            step=para.aux_weight_pc.q,
        )
    # All other descriptors currently pass through


def check_invalid_descriptor_param(alg: build.AnyAlgorithm) -> list:
    # if calibration is performed then base_estimator should be compat
    if isinstance(alg, Union[build.Mapie, build.CalibratedClassifierCVWithVA]):
        alg = alg.parameters.estimator
    # chemprop should have only chemprop descriptors
    if opt.isanyof(alg, build.AnyChemPropAlgorithm):
        return descriptors.SmilesBasedDescriptor.__args__
    #  all others should have non-chemprop descriptors
    else:
        return descriptors.AnyChemPropIncompatible.__args__
