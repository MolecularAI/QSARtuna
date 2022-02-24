from optunaz.utils.enums.configuration_enum import ConfigurationEnum


class InterfaceEnum:
    """This "Enum" serves as storage for the interface specifications."""

    # use the fixed names of the configuration enum
    _CE = ConfigurationEnum()

    # set the sets for the algorithms
    SKLEARN_SET = {_CE.ALGORITHMS_LASSO,
                   _CE.ALGORITHMS_RIDGE,
                   _CE.ALGORITHMS_LOGISTICREGRESSION,
                   _CE.ALGORITHMS_PLS,
                   _CE.ALGORITHMS_RF,
                   _CE.ALGORITHMS_SVC,
                   _CE.ALGORITHMS_SVR,
                   _CE.ALGORITHMS_ADABOOSTCLASSIFIER}
    XGBOOST_SET = {_CE.ALGORITHMS_XGBREGRESSOR}

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")


