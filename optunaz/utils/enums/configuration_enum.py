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

    # Avalon
    DESCRIPTORS_AVALON = "Avalon"
    DESCRIPTORS_AVALON_NBITS = "nBits"

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
    DESCRIPTORS_PHYSCHEM_RDKITNAMES = "rdkit_names"

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
    DESCRIPTORS_JAZZY_JAZZYNAMES = "jazzy_names"

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
    DESCRIPTORS_SMILES_AND_SI_AUX_WEIGHT_PC = "aux_weight_pc"

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
    SETTINGS_CROSS_VALIDATION = "cross_validation"
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

    # algorithm: AdaBoostClassifier
    ALGORITHMS_ADABOOSTCLASSIFIER = "AdaBoostClassifier"
    ALGORITHMS_ADABOOSTCLASSIFIER_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_ADABOOSTCLASSIFIER_LEARNING_RATE = "learning_rate"

    # algorithm: XGBregressor specific
    ALGORITHMS_XGBREGRESSOR = "XGBregressor"
    ALGORITHMS_XGBREGRESSOR_MAX_DEPTH = "max_depth"
    ALGORITHMS_XGBREGRESSOR_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_XGBREGRESSOR_LEARNING_RATE = "learning_rate"

    # algorithm: ProbabilisticRandomForest specific
    ALGORITHMS_PRF = "PRFClassifier"
    ALGORITHMS_PRF_MAX_FEATURES = "max_features"
    ALGORITHMS_PRF_MAX_DEPTH = "max_depth"
    ALGORITHMS_PRF_N_ESTIMATORS = "n_estimators"
    ALGORITHMS_PRF_MINPYSUMLEAF = "min_py_sum_leaf"
    ALGORITHMS_PRF_USE_PY_GINI = "use_py_gini"
    ALGORITHMS_PRF_USE_PY_LEAFS = "use_py_leafs"

    # algorithm: ChemProp specific
    ALGORITHMS_CHEMPROP = "BaseChemProp"
    ALGORITHMS_CHEMPROP_REGRESSOR = "ChemPropRegressor"
    ALGORITHMS_CHEMPROP_HYPEROPT_REGRESSOR = "ChemPropHyperoptRegressor"
    ALGORITHMS_CHEMPROP_CLASSIFIER = "ChemPropClassifier"
    ALGORITHMS_CHEMPROP_HYPEROPT_CLASSIFIER = "ChemPropHyperoptClassifier"
    ALGORITHMS_CHEMPROP_ACTIVATION = "activation"
    ALGORITHMS_CHEMPROP_AGGREGATION = "aggregation"
    ALGORITHMS_CHEMPROP_AGGREGATION_NORM = "aggregation_norm"
    ALGORITHMS_CHEMPROP_BATCH_SIZE = "batch_size"
    ALGORITHMS_CHEMPROP_DEPTH = "depth"
    ALGORITHMS_CHEMPROP_DROPOUT = "dropout"
    ALGORITHMS_CHEMPROP_EPOCHS = "epochs"
    ALGORITHMS_CHEMPROP_ENSEMBLE_SIZE = "ensemble_size"
    ALGORITHMS_CHEMPROP_FEATURES_GENERATOR = "features_generator"
    ALGORITHMS_CHEMPROP_FFN_HIDDEN_SIZE = "ffn_hidden_size"
    ALGORITHMS_CHEMPROP_FFN_NUM_LAYERS = "ffn_num_layers"
    ALGORITHMS_CHEMPROP_FRZN = "frzn"
    ALGORITHMS_CHEMPROP_FINAL_LR_RATIO_EXP = "final_lr_ratio_exp"
    ALGORITHMS_CHEMPROP_HIDDEN_SIZE = "hidden_size"
    ALGORITHMS_CHEMPROP_NUM_ITERS = "num_iters"
    ALGORITHMS_CHEMPROP_INIT_LR_RATIO_EXP = "init_lr_ratio_exp"
    ALGORITHMS_CHEMPROP_MAX_LR_EXP = "max_lr_exp"
    ALGORITHMS_CHEMPROP_PRETRAINED_MODEL = "pretrained_model"
    ALGORITHMS_CHEMPROP_SEARCH_PARAMETER_LEVEL = "search_parameter_level"
    ALGORITHMS_CHEMPROP_STARTUP_RANDOM_ITERS = "startup_random_iters"
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
    ALGORITHMS_CUSTOM_FILE = "preexisting_model"
    ALGORITHMS_CUSTOM_REFIT_MODEL = "refit_model"

    # algorithm: Mapie specific
    ALGORITHMS_MAPIE = "Mapie"
    ALGORITHMS_MAPIE_ALPHA = "mapie_alpha"
    ALGORITHMS_MAPIE_ESTIMATOR = "estimator"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
