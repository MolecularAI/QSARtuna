from enum import Enum

from apischema import serializer


class StudyUserAttrs(str, Enum):
    """Dict keys for User Parameters of Optuna Study objects."""
    OPTCONFIG = "optconfig"


class TrialUserAttrs(str, Enum):
    TRAIN_SCORES = "train_scores"
    TEST_SCORES = "test_scores"


class TrialParams(str, Enum):
    """Dict keys for Parameters of Optuna Trial objects."""
    DESCRIPTOR = "descriptor"
    ALGORITHM_NAME = "algorithm_name"


class MlflowLogParams(str, Enum):
    TRIAL_NUMBER = "trial_number"

