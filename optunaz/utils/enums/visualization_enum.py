class VisualizationEnum:
    """This "Enum" serves to store all the strings used to specify optional visualizations. Note, that validity
    checks are not performed, but referred to JSON Schema validations."""

    # general keywords
    # ---------
    VISUALIZATION_REGRESSOR = "regressor"
    VISUALIZATION_CLASSIFIER = "classifier"

    # top-level keywords
    # ---------
    VISUALIZATION = "visualization"
    VISUALIZATION_USE_XVFB = "use_xvfb"
    VISUALIZATION_OUTPUT_FOLDER = "output_folder"
    VISUALIZATION_FILE_FORMAT = "file_format"
    VISUALIZATION_PLOTS = "plots"

    # different plots
    # ---------
    VISUALIZATION_PLOTS_HISTORY = "plot_history"
    VISUALIZATION_PLOTS_CONTOUR = "plot_contour"
    VISUALIZATION_PLOTS_PARALLEL_COORDINATE = "plot_parallel_coordinate"
    VISUALIZATION_PLOTS_SLICE = "plot_slice"

    # internal optuna keywords
    # ---------
    OPTUNA_SYSTEM_ATTRS_NUMBER = "_number"
    OPTUNA_SYSTEM_ATTRS_INTERMEDIATE_VALUES = "intermediate_values"
    OPTUNA_SYSTEM_ATTRS_TRIAL_ID = "trial_id"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
