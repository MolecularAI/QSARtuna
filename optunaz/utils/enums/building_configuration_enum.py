from optunaz.utils.enums.configuration_enum import ConfigurationEnum


class BuildingConfigurationEnum(ConfigurationEnum):
    """This "Enum" serves to store all the strings used in parsing the building configurations. Note, that validity
    checks are not performed, but referred to JSON Schema validations."""

    # all that are general keywords
    GENERAL_HYPERPARAMETERS = "hyper_parameters"
    GENERAL_REGRESSOR = "regressor"
    GENERAL_CLASSIFIER = "classifier"

    # all that has to do with the generated metadata
    # ---------

    METADATA = "metadata"
    METADATA_BESTTRIAL = "best_trial"
    METADATA_BESTVALUE = "best_value"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
