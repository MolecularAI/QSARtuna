class ConfigurationEnum:
    """This "Enum" serves to store all the strings used in parsing all configurations. Note, that validity
    checks are not performed, but referred to JSON Schema validations."""

    # all that are general keywords
    GENERAL_DISABLED = "disabled"
    GENERAL_PARAMETERS = "parameters"
    GENERAL_NAME = "name"
    GENERAL_ALGORITHM_NAME = "algorithm_name"

    # all that has to do with the actual task
    # ---------
    TASK = "task"
    TASK_OPTIMIZATION = "optimization"
    TASK_BUILDING = "building"

    # all that has to do with data IO
    # ---------
    DATA = "data"
    DATA_INPUTCOLUMN = "input_column"
    DATA_RESPONSECOLUMN = "response_column"
    DATA_TRAINING = "training"
    DATA_TEST = "test"

    # all that has to do with descriptor calculation
    # ---------
    DESCRIPTORS = "descriptors"
    DESCRIPTORS_FEATURE_NAMES = "feature_names"

    # Avalon
    DESCRIPTORS_AVALON = "Avalon"
    DESCRIPTORS_AVALON_NBITS = "nBits"

    # AvalonCount
    DESCRIPTORS_AVALONCOUNT = "AvalonCount"
    DESCRIPTORS_AVALONCOUNT_NBITS = "nBits"

    # ECFP
    DESCRIPTORS_ECFP = "ECFP"
    DESCRIPTORS_ECFP_RADIUS = "radius"
    DESCRIPTORS_ECFP_NBITS = "nBits"

    # ECFP_counts
    DESCRIPTORS_ECFPCOUNTS = "ECFP_counts"
    DESCRIPTORS_ECFPCOUNTS_RADIUS = "radius"
    DESCRIPTORS_ECFPCOUNTS_USEFEATURES = "useFeatures"

    # PathFP
    DESCRIPTORS_PATHFP = "PathFP"
    DESCRIPTORS_PATHFP_MAXPATH = "maxPath"
    DESCRIPTORS_PATHFP_FPSIZE = "fpSize"

    # MACCS_keys
    DESCRIPTORS_MACCSKEYS = "MACCS_keys"

    # Physchem
    DESCRIPTORS_UNSC_PHYSCHEM = "UnscaledPhyschemDescriptors"
    DESCRIPTORS_PHYSCHEM = "PhyschemDescriptors"

    # AMORPROT
    DESCRIPTORS_AMORPROT = "AmorProtDescriptors"

    # MAPC
    DESCRIPTORS_UNSC_MAPC = "UnscaledMAPC"
    DESCRIPTORS_MAPC = "MAPC"
    DESCRIPTORS_MAPC_MAXRADIUS = "maxRadius"
    DESCRIPTORS_MAPC_NPERMUTATIONS = "nPermutations"

    # Jazzy
    DESCRIPTORS_UNSC_JAZZY = "UnscaledJazzyDescriptors"
    DESCRIPTORS_JAZZY = "JazzyDescriptors"
    DESCRIPTORS_JAZZY_JAZZYFILTERS = "jazzy_filters"
    DESCRIPTORS_JAZZY_EMBEDDINGTYPE = "embedding_type"
    DESCRIPTORS_JAZZY_EMBEDDINGMAXITERATIONDS = "embedding_max_iterations"
    DESCRIPTORS_JAZZY_EMBEDDINGSEED = "embedding_seed"

    # Mordred
    DESCRIPTORS_UNSC_MORDRED = "MordredDescriptors"
    DESCRIPTORS_MORDRED = "ScaledMordredDescriptors"
    DESCRIPTORS_MORDRED_DESCRIPTORSET = "ScaledMordredDescriptors"

    # Precomputed
    DESCRIPTORS_PRECOMPUTED = "PrecomputedDescriptorFromFile"
    DESCRIPTORS_PRECOMPUTED_FILE = "file"
    DESCRIPTORS_PRECOMPUTED_INPUT_COLUMNN = "input_column"
    DESCRIPTORS_PRECOMPUTED_RESPONSE_COLUMN = "response_column"

    # ZScales
    DESCRIPTORS_UNSC_ZSCALES = "UnscaledZScalesDescriptors"
    DESCRIPTORS_ZSCALES = "ZScalesDescriptors"

    # Smiles
    DESCRIPTORS_SMILES = "SmilesFromFile"
    DESCRIPTORS_SMILES_AND_SI = "SmilesAndSideInfoFromFile"
    DESCRIPTORS_SMILES_AND_SI_FILE = "file"
    DESCRIPTORS_SMILES_AND_SI_INPUT_COLUMN = "input_column"
    DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC = "y_aux_weight_pc"
    DESCRIPTORS_SMILES_AND_SI_Y_AUX_COLUMN = "y_aux_column"
    DESCRIPTORS_SMILES_AND_SI_X_AUX_COLUMN = "x_aux_column"

    # Scaled
    DESCRIPTORS_SCALED = "ScaledDescriptor"
    DESCRIPTORS_SCALED_DESCRIPTOR = "descriptor"
    DESCRIPTORS_SCALED_DESCRIPTOR_PARAMETERS = "parameters"

    # Composite
    DESCRIPTORS_COMPOSITE = "CompositeDescriptor"

    # all that has to do with general optimization parameters
    # ---------
    SETTINGS = "settings"
    SETTINGS_MODE = "mode"
    SETTINGS_MODE_REGRESSION = "regression"
    SETTINGS_MODE_CLASSIFICATION = "classification"
    SETTINGS_N_SPLITS = "n_splits"
    SETTINGS_N_REPLICATES = "n_replicates"
    SETTINGS_OPTIMISATION_SPLIT_STRATEGY = "optimization_split_strategy"
    SETTINGS_DIRECTION = "direction"
    SETTINGS_N_TRIALS = "n_trials"
    SETTINGS_N_JOBS = "n_jobs"
    SETTINGS_SHUFFLE = "shuffle"

    # all that has to do with algorithms, general keywords
    # ---------
    ALGORITHMS = "algorithms"
    ALGORITHMS_ESTIMATOR = "estimator"
    ALGORITHMS_LOW = "low"
    ALGORITHMS_HIGH = "high"
    ALGORITHMS_Q = "q"

    # different interfaces available
    ALGORITHMS_INTERFACE_SKLEARN = "sklearn"
    ALGORITHMS_INTERFACE_XGBOOST = "xgboost"

    # algorithm: RandomForest specific
    ALGORITHMS_RFREGRESSOR = "RandomForestRegressor"
    ALGORITHMS_RFCLASSIFIER = "RandomForestClassifier"
    ALGORITHMS_RF_MAX_FEATURES = "max_features"
    ALGORITHMS_RF_MAX_DEPTH = "max_depth"
    ALGORITHMS_RF_N_ESTIMATORS = "n_estimators"

    # algorithm: SVR
    ALGORITHMS_SVR = "SVR"
    ALGORITHMS_SVR_C = "C"
    ALGORITHMS_SVR_GAMMA = "gamma"

    # algorithm: SVC
    ALGORITHMS_SVC = "SVC"
    ALGORITHMS_SVC_C = "C"
    ALGORITHMS_SVC_GAMMA = "gamma"

    # algorithm: Lasso
    ALGORITHMS_LASSO = "Lasso"
    ALGORITHMS_LASSO_ALPHA = "alpha"
    ALGORITHMS_LASSO_MAX_ITER = "max_iter"
    ALGORITHMS_LASSO_TOL = "tol"

    # algorithm: KNeighbors
    ALGORITHMS_KNEIGHBORSCLASSIFIER = "KNeighborsClassifier"
    ALGORITHMS_KNEIGHBORSREGRESSOR = "KNeighborsRegressor"
    ALGORITHMS_KNEIGHBORS_N_NEIGHBORS = "n_neighbors"
    ALGORITHMS_KNEIGHBORS_METRIC = "metric"
    ALGORITHMS_KNEIGHBORS_WEIGHTS = "weights"

    # algorithm: Ridge
    ALGORITHMS_RIDGE = "Ridge"
    ALGORITHMS_RIDGE_ALPHA = "alpha"

    # algorithm: PLSRegression
    ALGORITHMS_PLSREGRESSION = "PLSRegression"
    ALGORITHMS_PLSREGRESSION_N_COMPONENTS = "n_components"

    # algorithm: LogisticRegression
    ALGORITHMS_LOGISTICREGRESSION = "LogisticRegression"
    ALGORITHMS_LOGISTICREGRESSION_SOLVER = "solver"
    ALGORITHMS_LOGISTICREGRESSION_C = "C"
    ALGORITHMS_LOGISTICREGRESSION_PENALTY = "penalty"

    # algorithm: AdaBoostClassifier
    ALGORITHMS_ADABOOSTCLASSIFIER = "AdaBoostClassifier"
    ALGORITHMS_ADABOOSTCLASSIFIER_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_ADABOOSTCLASSIFIER_LEARNING_RATE = "learning_rate"
    ALGORITHMS_ADABOOSTCLASSIFIER_MAX_DEPTH = "max_depth"

    # algorithm: CatBoostClassifier
    ALGORITHMS_CATBOOSTCLASSIFIER = "CatBoostClassifier"
    ALGORITHMS_CATBOOSTCLASSIFIER_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_CATBOOSTCLASSIFIER_DEPTH = "depth"
    ALGORITHMS_CATBOOSTCLASSIFIER_LEARNING_RATE = "learning_rate"
    ALGORITHMS_CATBOOSTCLASSIFIER_L2_LEAF_REG = "l2_leaf_reg"
    ALGORITHMS_CATBOOSTCLASSIFIER_RANDOM_STRENGTH = "random_strength"

    # algorithm: CatBoostRegressor
    ALGORITHMS_CATBOOSTREGRESSOR = "CatBoostRegressor"
    ALGORITHMS_CATBOOSTREGRESSOR_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_CATBOOSTREGRESSOR_DEPTH = "depth"
    ALGORITHMS_CATBOOSTREGRESSOR_LEARNING_RATE = "learning_rate"
    ALGORITHMS_CATBOOSTREGRESSOR_L2_LEAF_REG = "l2_leaf_reg"
    ALGORITHMS_CATBOOSTREGRESSOR_RANDOM_STRENGTH = "random_strength"

    # algorithm: XGBregressor specific
    ALGORITHMS_XGBREGRESSOR = "XGBregressor"
    ALGORITHMS_XGBREGRESSOR_MAX_DEPTH = "max_depth"
    ALGORITHMS_XGBREGRESSOR_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_XGBREGRESSOR_LEARNING_RATE = "learning_rate"
    ALGORITHMS_XGBREGRESSOR_SUBSAMPLE = "subsample"
    ALGORITHMS_XGBREGRESSOR_GAMMA = "gamma"
    ALGORITHMS_XGBREGRESSOR_COLSAMPLE_BYTREE = "colsample_bytree"

    # algorithm: XGBclassifier specific
    ALGORITHMS_XGBCLASSIFIER = "XGBclassifier"
    ALGORITHMS_XGBCLASSIFIER_MAX_DEPTH = "max_depth"
    ALGORITHMS_XGBCLASSIFIER_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_XGBCLASSIFIER_LEARNING_RATE = "learning_rate"
    ALGORITHMS_XGBCLASSIFIER_SUBSAMPLE = "subsample"
    ALGORITHMS_XGBCLASSIFIER_GAMMA = "gamma"
    ALGORITHMS_XGBCLASSIFIER_COLSAMPLE_BYTREE = "colsample_bytree"

    # algorithm: ProbabilisticRandomForest specific
    ALGORITHMS_PRF = "PRFClassifier"
    ALGORITHMS_PRF_MAX_FEATURES = "max_features"
    ALGORITHMS_PRF_MAX_DEPTH = "max_depth"
    ALGORITHMS_PRF_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_PRF_MINPYSUMLEAF = "min_py_sum_leaf"
    ALGORITHMS_PRF_USE_PY_GINI = "use_py_gini"
    ALGORITHMS_PRF_USE_PY_LEAFS = "use_py_leafs"

    # algorithm: TabPFN specific
    ALGORITHMS_TABPFN_CLASSIFIER = "TabPFNClassifier"
    ALGORITHMS_TABPFN_REGRESSOR = "TabPFNRegressor"
    ALGORITHMS_TABPFN_MAX_TIME = "max_time"
    ALGORITHMS_TABPFN_RANDOM_STATE = "random_state"
    ALGORITHMS_TABPFN_MAX_FEATS = "max_feats"
    ALGORITHMS_TABPFN_FEATURE_SELECTION = "feature_selection"
    ALGORITHMS_TABPFN_EVAL_METRIC = "eval_metric"

    # algorithm: FastProp specific
    ALGORITHMS_FASTPROP = "BaseFastProp"
    ALGORITHMS_FASTPROP_REGRESSOR = "FastPropRegressor"
    ALGORITHMS_FASTPROP_CLASSIFIER = "FastPropClassifier"
    ALGORITHMS_FASTPROP_FNN_LAYERS = "fnn_layers"
    ALGORITHMS_FASTPROP_LEARNING_RATE = "learning_rate"
    ALGORITHMS_FASTPROP_BATCH_SIZE = "batch_size"
    ALGORITHMS_FASTPROP_NUMBER_EPOCHS = "number_epochs"
    ALGORITHMS_FASTPROP_NUMBER_REPEATS = "number_repeats"
    ALGORITHMS_FASTPROP_TRAIN_SIZE = "train_size"
    ALGORITHMS_FASTPROP_VAL_SIZE = "val_size"
    ALGORITHMS_FASTPROP_TEST_SIZE = "test_size"
    ALGORITHMS_FASTPROP_RANDOM_SEED = "random_seed"
    ALGORITHMS_FASTPROP_HIDDEN_SIZE = "hidden_size"
    ALGORITHMS_FASTPROP_PATIENCE = "patience"

    # algorithm: ChemProp specific
    ALGORITHMS_CHEMPROP = "BaseChemProp"
    ALGORITHMS_CHEMPROP_REGRESSOR = "ChemPropRegressor"
    ALGORITHMS_CHEMPROP_REGRESSOR_PRETRAINED = "ChemPropRegressorPretrained"
    ALGORITHMS_CHEMPROP_CLASSIFIER = "ChemPropClassifier"
    ALGORITHMS_CHEMPROP_ACTIVATION = "activation"
    ALGORITHMS_CHEMPROP_AGGREGATION = "aggregation"
    ALGORITHMS_CHEMPROP_AGGREGATION_NORM = "aggregation_norm"
    ALGORITHMS_CHEMPROP_BATCH_SIZE = "batch_size"
    ALGORITHMS_CHEMPROP_BATCH_NORM = "batch_norm"
    ALGORITHMS_CHEMPROP_DEPTH = "depth"
    ALGORITHMS_CHEMPROP_DROPOUT = "dropout"
    ALGORITHMS_CHEMPROP_TRIAL_DROPOUT = "trial_dropout"
    ALGORITHMS_CHEMPROP_EPOCHS = "epochs"
    ALGORITHMS_CHEMPROP_PATIENCE = "patience"
    ALGORITHMS_CHEMPROP_ENSEMBLE_SIZE = "ensemble_size"
    ALGORITHMS_CHEMPROP_MOLECULE_FEATURIZERS = "molecule_featurizers"
    ALGORITHMS_CHEMPROP_FFN_HIDDEN_DIM = "ffn_hidden_dim"
    ALGORITHMS_CHEMPROP_FFN_NUM_LAYERS = "ffn_num_layers"
    ALGORITHMS_CHEMPROP_FRZN = "frzn"
    ALGORITHMS_CHEMPROP_FINAL_LR_RATIO = "final_lr_ratio"
    ALGORITHMS_CHEMPROP_MESSAGE_BIAS = "message_bias"
    ALGORITHMS_CHEMPROP_MESSAGE_HIDDEN_DIM = "message_hidden_dim"
    ALGORITHMS_CHEMPROP_LOSS_FUNCTION = "loss_function"
    ALGORITHMS_CHEMPROP_NUM_ITERS = "num_iters"
    ALGORITHMS_CHEMPROP_INIT_LR_RATIO = "init_lr_ratio"
    ALGORITHMS_CHEMPROP_MAX_LR = "max_lr"
    ALGORITHMS_CHEMPROP_PRETRAINED_MODEL = "pretrained_model"
    ALGORITHMS_CHEMPROP_SEARCH_PARAMETER_LEVEL = "search_parameter_level"
    ALGORITHMS_CHEMPROP_STARTUP_RANDOM_ITERS = "startup_random_iters"
    ALGORITHMS_CHEMPROP_UNDIRECTED = "undirected"
    ALGORITHMS_CHEMPROP_WARMUP_EPOCHS_RATIO = "warmup_epochs_ratio"

    # algorithm: CalibratedClassifierCV specific
    ALGORITHMS_CALIBRATEDCLASSIFIERCV = "CalibratedClassifierCVWithVA"
    ALGORITHMS_CALIBRATEDCLASSIFIERCV_ENSEMBLE = "ensemble"
    ALGORITHMS_CALIBRATEDCLASSIFIERCV_ESTIMATOR = "estimator"
    ALGORITHMS_CALIBRATEDCLASSIFIERCV_METHOD = "method"
    ALGORITHMS_CALIBRATEDCLASSIFIERCV_N_FOLDS = "n_folds"
    ALGORITHMS_CALIBRATEDCLASSIFIERCV_PARAMS = "calibrated_params"

    # algorithm: CustomModels
    ALGORITHMS_CUSTOMREGRESSIONMODEL = "CustomRegressionModel"
    ALGORITHMS_CUSTOMCLASSIFIERMODEL = "CustomClassifierModel"
    ALGORITHMS_CUSTOM_FILE = "model_file"
    ALGORITHMS_CUSTOM_REFIT_MODEL = "refit_model"

    # algorithm: Mapie specific
    ALGORITHMS_MAPIE_REGRESSOR = "MapieRegressor"
    ALGORITHMS_MAPIE_CLASSIFIER = "MapieClassifier"
    ALGORITHMS_MAPIE_ALPHA = "mapie_alpha"
    ALGORITHMS_MAPIE_TEST_SIZE = "test_size"
    ALGORITHMS_MAPIE_N_FOLDS = "n_folds"
    ALGORITHMS_MAPIE_ESTIMATOR = "estimator"
    ALGORITHMS_MAPIE_RANDOM_STATE = "random_state"

    # try to find the internal value and return
    def __getattr__(self, name):
        if hasattr(self.__class__, name):
            return super().__getattr__(name)
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise AttributeError("No changes allowed.")
