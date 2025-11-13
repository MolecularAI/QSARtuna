import json
from functools import partial
from itertools import combinations
from typing import Union

import numpy as np
from apischema import deserialize, serialize
from joblib import Memory, effective_n_jobs
from optuna import Study
from optuna.trial import FrozenTrial

import optunaz.config.buildconfig as build
import optunaz.config.optconfig as opt
import optunaz.descriptors as descriptors
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.descriptors import MolDescriptor
from optunaz.utils import mkdict
from optunaz.utils.enums import StudyUserAttrs, TrialParams
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
    y_aux_weight_pc = trial.params.get(_CE.DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC, 100)

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
        ] = y_aux_weight_pc

    algorithm_dict = {
        _CE.GENERAL_NAME: trial.params.get(_CE.GENERAL_ALGORITHM_NAME),
        _CE.GENERAL_PARAMETERS: mkdict(
            {
                **trial.params,
                **calibrated_params,
                **{
                    _CE.ALGORITHMS_ESTIMATOR: base_estimator,
                    _CE.DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC: y_aux_weight_pc,
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
            n_splits=optconfig.settings.n_splits,
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
            tracking_rest_endpoint=optconfig.settings.tracking_rest_endpoint,
        ),
    )


def encode_name(CEname, hash=hash):
    """Encode the parameter names with a hash to enable multi-"""
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
            log=True,
        )
        max_depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_ADABOOSTCLASSIFIER_MAX_DEPTH),
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        return build.AdaBoostClassifier.new(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
    elif isinstance(alg, opt.CatBoostClassifier):
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTCLASSIFIER_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        learning_rate = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTCLASSIFIER_LEARNING_RATE),
            low=para.learning_rate.low,
            high=para.learning_rate.high,
            log=True,
        )
        depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTCLASSIFIER_DEPTH),
            low=para.depth.low,
            high=para.depth.high,
        )
        l2_leaf_reg = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTCLASSIFIER_L2_LEAF_REG),
            low=para.l2_leaf_reg.low,
            high=para.l2_leaf_reg.high,
            log=True,
        )
        random_strength = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTCLASSIFIER_RANDOM_STRENGTH),
            low=para.random_strength.low,
            high=para.random_strength.high,
            log=True,
        )
        return build.CatBoostClassifier.new(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
        )
    elif isinstance(alg, opt.CatBoostRegressor):
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTREGRESSOR_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        learning_rate = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTREGRESSOR_LEARNING_RATE),
            low=para.learning_rate.low,
            high=para.learning_rate.high,
            log=True,
        )
        depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTREGRESSOR_DEPTH),
            low=para.depth.low,
            high=para.depth.high,
        )
        l2_leaf_reg = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTREGRESSOR_L2_LEAF_REG),
            low=para.l2_leaf_reg.low,
            high=para.l2_leaf_reg.high,
            log=True,
        )
        random_strength = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CATBOOSTREGRESSOR_RANDOM_STRENGTH),
            low=para.random_strength.low,
            high=para.random_strength.high,
            log=True,
        )
        return build.CatBoostRegressor.new(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
        )
    elif isinstance(alg, opt.Lasso):
        alpha = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_LASSO_ALPHA),
            low=para.alpha.low,
            high=para.alpha.high,
            log=True
        )
        max_iter = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_LASSO_MAX_ITER),
            low=para.max_iter.low,
            high=para.max_iter.high,
            step=para.max_iter.step
        )
        tol = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_LASSO_TOL),
            low=para.tol.low,
            high=para.tol.high,
            log=True
        )
        return build.Lasso.new(alpha=alpha, max_iter=max_iter, tol=tol)
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
        penalty = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_LOGISTICREGRESSION_PENALTY),
            choices=para.penalty,
        )
        return build.LogisticRegression.new(solver=solver, C=lg_c, penalty=penalty)
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
            log=True,
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
    elif isinstance(alg, opt.XGBClassifier):
        max_depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_XGBCLASSIFIER_MAX_DEPTH),
            low=para.max_depth.low,
            high=para.max_depth.high,
        )
        n_estimators = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_XGBCLASSIFIER_N_ESTIMATORS),
            low=para.n_estimators.low,
            high=para.n_estimators.high,
        )
        learning_rate = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBCLASSIFIER_LEARNING_RATE),
            low=para.learning_rate.low,
            high=para.learning_rate.high,
            log=True,
        )
        subsample = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBCLASSIFIER_SUBSAMPLE),
            low=para.subsample.low,
            high=para.subsample.high,
        )
        gamma = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBCLASSIFIER_GAMMA),
            low=para.gamma.low,
            high=para.gamma.high,
            log=True,
        )
        colsample_bytree = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBCLASSIFIER_COLSAMPLE_BYTREE),
            low=para.colsample_bytree.low,
            high=para.colsample_bytree.high,
        )
        return build.XGBClassifier.new(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            gamma=gamma,
            colsample_bytree=colsample_bytree
        )
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
            log=True,
        )
        subsample = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBREGRESSOR_SUBSAMPLE),
            low=para.subsample.low,
            high=para.subsample.high,
        )
        gamma = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBREGRESSOR_GAMMA),
            low=para.gamma.low,
            high=para.gamma.high,
            log=True,
        )
        colsample_bytree = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_XGBREGRESSOR_COLSAMPLE_BYTREE),
            low=para.colsample_bytree.low,
            high=para.colsample_bytree.high,
        )
        return build.XGBRegressor.new(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            gamma=gamma,
            colsample_bytree=colsample_bytree
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
    elif isinstance(alg, opt.TabPFNClassifier):
        max_time = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_MAX_TIME),
            low=para.max_time,
            high=para.max_time,
        )
        random_state = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_RANDOM_STATE),
            low=para.random_state,
            high=para.random_state,
        )
        max_feats = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_MAX_FEATS),
            low=para.max_feats,
            high=para.max_feats,
        )
        feature_selection = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_FEATURE_SELECTION),
            choices=para.feature_selection,
        )
        eval_metric = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_EVAL_METRIC),
            choices=para.eval_metric,
        )
        return build.TabPFNClassifier.new(
            max_time=max_time,
            random_state=random_state,
            max_feats=max_feats,
            feature_selection=feature_selection,
            eval_metric=eval_metric,
        )
    elif isinstance(alg, opt.TabPFNRegressor):
        max_time = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_MAX_TIME),
            low=para.max_time,
            high=para.max_time,
        )
        random_state = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_RANDOM_STATE),
            low=para.random_state,
            high=para.random_state,
        )
        max_feats = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_MAX_FEATS),
            low=para.max_feats,
            high=para.max_feats,
        )
        feature_selection = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_FEATURE_SELECTION),
            choices=para.feature_selection,
        )
        eval_metric = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_TABPFN_EVAL_METRIC),
            choices=para.eval_metric,
        )
        return build.TabPFNRegressor.new(
            max_time=max_time,
            random_state=random_state,
            max_feats=max_feats,
            feature_selection=feature_selection,
            eval_metric=eval_metric,
        )
    elif isinstance(alg, opt.FastPropClassifier):
        hidden_size = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_HIDDEN_SIZE),
            low=para.hidden_size.low,
            high=para.hidden_size.high,
            step=para.hidden_size.step,
        )
        fnn_layers = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_FNN_LAYERS),
            low=para.fnn_layers.low,
            high=para.fnn_layers.high,
            step=para.fnn_layers.step,
        )
        learning_rate = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_LEARNING_RATE),
            low=para.learning_rate.low,
            high=para.learning_rate.high,
            log=True,
        )
        number_epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_NUMBER_EPOCHS),
            low=para.number_epochs,
            high=para.number_epochs,
        )
        number_repeats = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_NUMBER_REPEATS),
            low=para.number_repeats,
            high=para.number_repeats,
        )
        patience = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_PATIENCE),
            low=para.patience,
            high=para.patience,
        )
        batch_size = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_BATCH_SIZE),
            choices=para.batch_size,
        )
        train_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_TRAIN_SIZE),
            low=para.train_size,
            high=para.train_size,
        )
        val_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_VAL_SIZE),
            low=para.val_size,
            high=para.val_size,
        )
        test_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_TEST_SIZE),
            low=para.test_size,
            high=para.test_size,
        )
        random_seed = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_RANDOM_SEED),
            low=para.random_seed,
            high=para.random_seed,
        )
        return build.FastPropClassifier.new(
            batch_size=batch_size,
            number_epochs=number_epochs,
            number_repeats=number_repeats,
            patience=patience,
            hidden_size=hidden_size,
            fnn_layers=fnn_layers,
            learning_rate=learning_rate,
            random_seed=random_seed,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
    elif isinstance(alg, opt.FastPropRegressor):
        hidden_size = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_HIDDEN_SIZE),
            low=para.hidden_size.low,
            high=para.hidden_size.high,
            step=para.hidden_size.step,
        )
        fnn_layers = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_FNN_LAYERS),
            low=para.fnn_layers.low,
            high=para.fnn_layers.high,
            step=para.fnn_layers.step,
        )
        learning_rate = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_LEARNING_RATE),
            low=para.learning_rate.low,
            high=para.learning_rate.high,
            log=True,
        )
        number_epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_NUMBER_EPOCHS),
            low=para.number_epochs,
            high=para.number_epochs,
        )
        number_repeats = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_NUMBER_REPEATS),
            low=para.number_repeats,
            high=para.number_repeats,
        )
        patience = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_PATIENCE),
            low=para.patience,
            high=para.patience,
        )
        batch_size = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_BATCH_SIZE),
            choices=para.batch_size,
        )
        train_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_TRAIN_SIZE),
            low=para.train_size,
            high=para.train_size,
        )
        val_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_VAL_SIZE),
            low=para.val_size,
            high=para.val_size,
        )
        test_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_TEST_SIZE),
            low=para.test_size,
            high=para.test_size,
        )
        random_seed = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_FASTPROP_RANDOM_SEED),
            low=para.random_seed,
            high=para.random_seed,
        )
        return build.FastPropRegressor.new(
            batch_size=batch_size,
            number_epochs=number_epochs,
            number_repeats=number_repeats,
            patience=patience,
            hidden_size=hidden_size,
            fnn_layers=fnn_layers,
            learning_rate=learning_rate,
            random_seed=random_seed,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
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
        aggregation_norm = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_AGGREGATION_NORM),
            low=para.aggregation_norm.low,
            high=para.aggregation_norm.high,
            step=para.aggregation_norm.step,
        )
        batch_size = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_BATCH_SIZE),
            choices=para.batch_size,
        )
        batch_norm = bool(
            trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_BATCH_NORM),
                choices=para.batch_norm,
            )
        )
        depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DEPTH),
            low=para.depth.low,
            high=para.depth.high,
            step=para.depth.step,
        )
        dropout = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DROPOUT),
            choices=[0.0] * 8 + list(np.arange(0.05, 0.45, 0.05))
            if para.trial_dropout
            else [0.0],
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
        patience = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_PATIENCE),
            low=para.patience,
            high=para.patience,
        )
        if para.molecule_featurizers:
            molecule_featurizers_combinations = [
                list(comb)
                for r in range(1, len(para.molecule_featurizers) + 1)
                for comb in combinations(para.molecule_featurizers, r)
            ]
            molecule_featurizers = trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MOLECULE_FEATURIZERS),
                choices=molecule_featurizers_combinations,
            )
        else:
            molecule_featurizers = None
        ffn_hidden_dim = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_HIDDEN_DIM),
            low=para.ffn_hidden_dim.low,
            high=para.ffn_hidden_dim.high,
            step=para.ffn_hidden_dim.step,
        )
        ffn_num_layers = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_NUM_LAYERS),
            low=para.ffn_num_layers.low,
            high=para.ffn_num_layers.high,
            step=para.ffn_num_layers.step,
        )
        final_lr_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FINAL_LR_RATIO),
            low=para.final_lr_ratio.low,
            high=para.final_lr_ratio.high,
            log=True,
        )
        message_bias = bool(
            trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MESSAGE_BIAS),
                choices=para.message_bias,
            )
        )
        message_hidden_dim = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MESSAGE_HIDDEN_DIM),
            low=para.message_hidden_dim.low,
            high=para.message_hidden_dim.high,
            step=para.message_hidden_dim.step,
        )
        init_lr_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_INIT_LR_RATIO),
            low=para.init_lr_ratio.low,
            high=para.init_lr_ratio.high,
            log=True,
        )
        max_lr = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MAX_LR),
            low=para.max_lr.low,
            high=para.max_lr.high,
            log=True,
        )
        warmup_epochs_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_WARMUP_EPOCHS_RATIO),
            low=para.warmup_epochs_ratio.low,
            high=para.warmup_epochs_ratio.high,
            step=para.warmup_epochs_ratio.step,
        )
        loss_function = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_LOSS_FUNCTION),
            choices=para.loss_function,
        )
        undirected = bool(
            trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_UNDIRECTED),
                choices=para.undirected,
            )
        )
        return build.ChemPropClassifier.new(
            activation=activation,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            batch_size=batch_size,
            batch_norm=batch_norm,
            depth=depth,
            dropout=dropout,
            ensemble_size=ensemble_size,
            epochs=epochs,
            patience=patience,
            molecule_featurizers=molecule_featurizers,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            final_lr_ratio=final_lr_ratio,
            message_hidden_dim=message_hidden_dim,
            message_bias=message_bias,
            loss_function=loss_function,
            init_lr_ratio=init_lr_ratio,
            max_lr=max_lr,
            warmup_epochs_ratio=warmup_epochs_ratio,
            undirected=undirected,
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
        aggregation_norm = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_AGGREGATION_NORM),
            low=para.aggregation_norm.low,
            high=para.aggregation_norm.high,
            step=para.aggregation_norm.step,
        )
        batch_size = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_BATCH_SIZE),
            choices=para.batch_size,
        )
        batch_norm = bool(
            trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_BATCH_NORM),
                choices=para.batch_norm,
            )
        )
        depth = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DEPTH),
            low=para.depth.low,
            high=para.depth.high,
            step=para.depth.step,
        )
        dropout = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_DROPOUT),
            choices=[0.0] * 8 + list(np.arange(0.05, 0.45, 0.05))
            if para.trial_dropout
            else [0.0],
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
        patience = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_PATIENCE),
            low=para.patience,
            high=para.patience,
        )
        if para.molecule_featurizers:
            molecule_featurizers_combinations = [
                list(comb)
                for r in range(1, len(para.molecule_featurizers) + 1)
                for comb in combinations(para.molecule_featurizers, r)
            ]
            molecule_featurizers = trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MOLECULE_FEATURIZERS),
                choices=molecule_featurizers_combinations,
            )
        else:
            molecule_featurizers = None
        ffn_hidden_dim = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_HIDDEN_DIM),
            low=para.ffn_hidden_dim.low,
            high=para.ffn_hidden_dim.high,
            step=para.ffn_hidden_dim.step,
        )
        ffn_num_layers = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FFN_NUM_LAYERS),
            low=para.ffn_num_layers.low,
            high=para.ffn_num_layers.high,
            step=para.ffn_num_layers.step,
        )
        final_lr_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FINAL_LR_RATIO),
            low=para.final_lr_ratio.low,
            high=para.final_lr_ratio.high,
            log=True,
        )
        message_bias = bool(
            trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MESSAGE_BIAS),
                choices=para.message_bias,
            )
        )
        message_hidden_dim = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MESSAGE_HIDDEN_DIM),
            low=para.message_hidden_dim.low,
            high=para.message_hidden_dim.high,
            step=para.message_hidden_dim.step,
        )
        init_lr_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_INIT_LR_RATIO),
            low=para.init_lr_ratio.low,
            high=para.init_lr_ratio.high,
            log=True,
        )
        max_lr = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_MAX_LR),
            low=para.max_lr.low,
            high=para.max_lr.high,
            log=True,
        )
        warmup_epochs_ratio = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_WARMUP_EPOCHS_RATIO),
            low=para.warmup_epochs_ratio.low,
            high=para.warmup_epochs_ratio.high,
            step=para.warmup_epochs_ratio.step,
        )
        loss_function = trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_LOSS_FUNCTION),
            choices=para.loss_function,
        )
        undirected = bool(
            trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_UNDIRECTED),
                choices=para.undirected,
            )
        )
        return build.ChemPropRegressor.new(
            activation=activation,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            batch_size=batch_size,
            batch_norm=batch_norm,
            depth=depth,
            dropout=dropout,
            ensemble_size=ensemble_size,
            epochs=epochs,
            patience=patience,
            molecule_featurizers=molecule_featurizers,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            final_lr_ratio=final_lr_ratio,
            message_bias=message_bias,
            message_hidden_dim=message_hidden_dim,
            loss_function=loss_function,
            init_lr_ratio=init_lr_ratio,
            max_lr=max_lr,
            warmup_epochs_ratio=warmup_epochs_ratio,
            undirected=undirected,
        )
    elif isinstance(alg, opt.ChemPropRegressorPretrained):
        if para.frzn is not None:
            frzn = trial.suggest_categorical(
                name=_encode_name(_CE.ALGORITHMS_CHEMPROP_FRZN), choices=para.frzn
            )
        else:
            frzn = None
        epochs = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_EPOCHS),
            low=para.epochs,
            high=para.epochs,
        )
        patience = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CHEMPROP_PATIENCE),
            low=para.patience,
            high=para.patience,
        )
        trial.set_user_attr(
            key=_CE.ALGORITHMS_CHEMPROP_PRETRAINED_MODEL, value=para.pretrained_model
        )

        return build.ChemPropRegressorPretrained.new(
            epochs=epochs,
            frzn=frzn,
            patience=patience,
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
    elif isinstance(alg, opt.MapieRegressor):
        mapie_alpha = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_ALPHA),
            low=para.mapie_alpha,
            high=para.mapie_alpha,
        )
        test_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_TEST_SIZE),
            low=para.test_size,
            high=para.test_size,
        )
        random_state = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_RANDOM_STATE),
            low=para.random_state,
            high=para.random_state,
        )
        n_folds = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_N_FOLDS),
            low=para.n_folds,
            high=para.n_folds,
        )
        estimator = suggest_alg_params(trial, para.estimator)
        trial.set_user_attr(
            key=_CE.ALGORITHMS_MAPIE_ESTIMATOR, value=serialize(estimator)
        )

        return build.MapieRegressor.new(
            estimator=estimator,
            mapie_alpha=mapie_alpha,
            n_folds=n_folds,
            test_size=test_size,
            random_state=random_state,
        )
    elif isinstance(alg, opt.MapieClassifier):
        mapie_alpha = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_ALPHA),
            low=para.mapie_alpha,
            high=para.mapie_alpha,
        )
        test_size = trial.suggest_float(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_TEST_SIZE),
            low=para.test_size,
            high=para.test_size,
        )
        random_state = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_RANDOM_STATE),
            low=para.random_state,
            high=para.random_state,
        )
        n_folds = trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_MAPIE_N_FOLDS),
            low=para.n_folds,
            high=para.n_folds,
        )
        estimator = suggest_alg_params(trial, para.estimator)
        trial.set_user_attr(
            key=_CE.ALGORITHMS_MAPIE_ESTIMATOR, value=serialize(estimator)
        )

        return build.MapieClassifier.new(
            estimator=estimator,
            mapie_alpha=mapie_alpha,
            n_folds=n_folds,
            test_size=test_size,
            random_state=random_state,
        )
    elif isinstance(alg, opt.CustomRegressionModel):
        trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_FILE),
            choices=[para.model_file],
        )
        trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_REFIT_MODEL),
            low=para.refit_model,
            high=para.refit_model,
        )
        return build.CustomRegressionModel.new(
            model_file=para.model_file,
            refit_model=para.refit_model,
        )
    elif isinstance(alg, opt.CustomClassificationModel):
        trial.suggest_categorical(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_FILE),
            choices=[para.model_file],
        )
        trial.suggest_int(
            name=_encode_name(_CE.ALGORITHMS_CUSTOM_REFIT_MODEL),
            low=para.refit_model,
            high=para.refit_model,
        )
        return build.CustomClassificationModel.new(
            model_file=para.model_file,
            refit_model=para.refit_model,
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
            low=para.y_aux_weight_pc.low,
            high=para.y_aux_weight_pc.high,
            step=para.y_aux_weight_pc.step,
        )
    return None
    # All other descriptors currently pass through


def suggest_n_jobs(n_jobs, alg: opt.AnyAlgorithm) -> int:
    """Suggest the number of cores for parallel processing based on the algorithm."""
    if (
        isinstance(alg, build.AvoidNestedParallelism.__args__)
        or hasattr(alg.parameters, "estimator")
        and isinstance(
            alg.parameters.estimator,
            build.AvoidNestedParallelism.__args__,
        )
    ):
        return 1
    return effective_n_jobs(n_jobs)


def check_invalid_descriptor_param(alg: build.AnyAlgorithm) -> list:
    # if calibration is performed then base_estimator should be compat
    if isinstance(
        alg,
        Union[
            build.MapieRegressor,
            build.MapieClassifier,
            build.CalibratedClassifierCVWithVA,
        ],
    ):
        alg = alg.parameters.estimator
    # chemprop should have only chemprop descriptors
    if isinstance(alg, build.AnyChemPropAlgorithm.__args__):
        return descriptors.SmilesBasedDescriptor.__args__
    #  all others should have non-chemprop descriptors
    else:
        return descriptors.AnyChemPropIncompatible.__args__
