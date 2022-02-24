from optunaz.utils.enums.configuration_enum import ConfigurationEnum


class OptimizationConfigurationEnum(ConfigurationEnum):
    """This "Enum" serves to store all the strings used in parsing the optimization configurations. Note, that validity
       checks are not performed, but referred to JSON Schema validations."""

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")


