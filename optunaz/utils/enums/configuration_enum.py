
class ConfigurationEnum:
    """This "Enum" serves to store all the strings used in parsing all configurations. Note, that validity
       checks are not performed, but referred to JSON Schema validations."""

    # all that are general keywords
    GENERAL_DISABLED = "disabled"
    GENERAL_PARAMETERS = "parameters"

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

    # ECFP
    DESCRIPTORS_ECFP = "ECFP"
    DESCRIPTORS_ECFP_RADIUS = "radius"
    DESCRIPTORS_ECFP_NBITS = "nBits"

    # ECFP_counts
    DESCRIPTORS_ECFPCOUNTS = "ECFP_counts"
    DESCRIPTORS_ECFPCOUNTS_RADIUS = "radius"
    DESCRIPTORS_ECFPCOUNTS_USEFEATURES = "useFeatures"

    # Avalon
    DESCRIPTORS_AVALON = "Avalon"
    DESCRIPTORS_AVALON_NBITS = "nBits"

    # MACCS_keys
    DESCRIPTORS_MACCSKEYS = "MACCS_keys"

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
    ALGORITHMS_LOW = "low"
    ALGORITHMS_HIGH = "high"

    # different interfaces available
    ALGORITHMS_INTERFACE_SKLEARN = "sklearn"
    ALGORITHMS_INTERFACE_XGBOOST = "xgboost"

    # algorithm: RandomForest specific
    ALGORITHMS_RF = "RandomForest"
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

    # algorithm: Ridge
    ALGORITHMS_RIDGE = "Ridge"
    ALGORITHMS_RIDGE_ALPHA = "alpha"

    # algorithm: PLS
    ALGORITHMS_PLS = "PLS"
    ALGORITHMS_PLS_N_COMPONENTS = "n_components"

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

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")


