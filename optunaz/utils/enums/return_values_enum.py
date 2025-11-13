class SklearnReturnValueEnum:
    """This "Enum" serves as storage for the return value keys using "sklearn"."""

    # Function: "cross_validate()"
    # ---------
    CROSS_VALIDATE_FIT_TIME = "fit_time"
    CROSS_VALIDATE_SCORE_TIME = "score_time"
    CROSS_VALIDATE_TEST_SCORE = "test_score"
    CROSS_VALIDATE_TRAIN_SCORE = "train_score"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")


class XGBoostReturnValueEnum:
    """This "Enum" serves as storage for the return value keys using "XGBoost"."""

    # Function: "cross_validate()"
    # ---------
    CROSS_VALIDATE_FIT_TIME = "fit_time"
    CROSS_VALIDATE_SCORE_TIME = "score_time"
    CROSS_VALIDATE_TEST_SCORE = "test_score"
    CROSS_VALIDATE_TRAIN_SCORE = "train_score"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
