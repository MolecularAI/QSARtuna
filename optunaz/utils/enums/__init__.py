from enum import Enum


class StudyUserAttrs(str, Enum):
    """Dict keys for User Parameters of Optuna Study objects."""

    OPTCONFIG = "optconfig"


class TrialUserAttrs(str, Enum):
    """Dict keys for User Parameters of Optuna Trial objects."""

    TRAIN_SCORES = "train_scores"
    TEST_SCORES = "test_scores"


class TrialParams(str, Enum):
    """Dict keys for Parameters of Optuna Trial objects."""

    DESCRIPTOR = "descriptor"
    ALGORITHM_NAME = "algorithm_name"
    ALGORITHM_HASH = "algorithm_hash"


class MlflowLogParams(str, Enum):
    """Dict keys for Parameters of MLflow Logs."""

    TRIAL_NUMBER = "trial_number"
