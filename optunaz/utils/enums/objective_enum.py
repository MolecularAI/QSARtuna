class ObjectiveEnum:
    """This "Enum" serves to store all the strings appearing in the objective function (and its return values)."""

    # user attributes
    # ---------
    ATTRIBUTE_TRIAL_TRAIN_SCORE = "train_score"

    # extra column names
    # ---------
    EXTRA_COLUMN_BESTHIT = "bestHit"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
