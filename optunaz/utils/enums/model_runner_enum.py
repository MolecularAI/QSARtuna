
class ModelRunnerDataframeEnum:
    """This "Enum" serves as storage for the column names used in the dataframe construction in the ModelRunner."""

    # obligatory
    # ---------
    Y_PRED = "y_pred"
    Y_TRUE = "y_true"
    SET = "set"
    TRAIN = "train"
    TEST = "test"

    # optional
    # ---------
    SMILES = "smiles"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")