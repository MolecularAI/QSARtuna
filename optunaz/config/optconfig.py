from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional, List, Iterable, Type, Any

from apischema import schema, type_name
from typing_extensions import Literal

from optunaz.config import (
    ModelMode,
    OptimizationDirection,
    Algorithm as GenericAlg,
    Visualization,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    MolDescriptor,
    ScaledDescriptor,
    UnfittedSklearnScaler,
    CompositeDescriptor,
)

from optunaz.utils import mkdict


class ClassificationScore(str, Enum):
    ACCURACY = "accuracy"
    AVERAGE_PRECISION = "average_precision"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1 = "f1"
    F1_MACRO = "f1_macro"
    F1_MICRO = "f1_micro"
    F1_WEIGHTED = "f1_weighted"
    JACCARD = "jaccard"
    JACCARD_MACRO = "jaccard_macro"
    JACCARD_MICRO = "jaccard_micro"
    JACCARD_WEIGHTED = "jaccard_weighted"
    PRECISION = "precision"
    PRECISION_MACRO = "precision_macro"
    PRECISION_MICRO = "precision_micro"
    PRECISION_WEIGHTED = "precision_weighted"
    RECALL = "recall"
    RECALL_MACRO = "recall_macro"
    RECALL_MICRO = "recall_micro"
    RECALL_WEIGHTED = "recall_weighted"
    ROC_AUC = "roc_auc"


class RegressionScore(str, Enum):
    EXPLAINED_VARIANCE = "explained_variance"
    MAX_ERROR = "max_error"
    NEG_MEAN_ABSOLUTE_ERROR = "neg_mean_absolute_error"
    NEG_MEAN_SQUARED_ERROR = "neg_mean_squared_error"
    NEG_MEDIAN_ABSOLUTE_ERROR = "neg_median_absolute_error"
    R2 = "r2"


class Algorithm(GenericAlg):
    pass


@dataclass
class AdaBoostClassifier(Algorithm):
    """AdaBoost Classifier."""

    @type_name("AdaBoostClassifierParams")
    @dataclass
    class Parameters:
        @dataclass
        class AdaBoostClassifierParametersNEstimators:
            low: int = field(default=3, metadata=schema(min=1))
            high: int = field(default=100, metadata=schema(min=1))

        @dataclass
        class AdaBoostClassifierParametersLearningRate:
            low: float = field(default=1.0, metadata=schema(min=0.0001))
            high: float = field(default=1.0, metadata=schema(min=0.001))

        n_estimators: AdaBoostClassifierParametersNEstimators = (
            AdaBoostClassifierParametersNEstimators()
        )
        learning_rate: AdaBoostClassifierParametersLearningRate = (
            AdaBoostClassifierParametersLearningRate()
        )

    name: Literal["AdaBoostClassifier"]
    parameters: Parameters


@dataclass
class Lasso(Algorithm):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)."""

    @type_name("LassoParams")
    @dataclass
    class Parameters:
        @dataclass
        class LassoParametersAlpha:
            low: float = field(default=0.0, metadata=schema(min=0))
            high: float = field(default=2.0, metadata=schema(min=0))

        alpha: LassoParametersAlpha = LassoParametersAlpha()

    name: Literal["Lasso"]
    parameters: Parameters


@dataclass
class LogisticRegression(Algorithm):
    """Logistic Regression (aka logit, MaxEnt) classifier."""

    @type_name("LogisticRegressionParams")
    @dataclass
    class Parameters:
        @dataclass
        class LogisticRegressionParametersParameterC:
            low: float = field(default=1.0, metadata=schema(min=0.001))
            high: float = field(default=1.0, metadata=schema(max=1000))

        solver: List[str] = field(
            default_factory=lambda: ["newton-cg", "lbfgs", "sag", "saga"]
        )
        C: LogisticRegressionParametersParameterC = (
            LogisticRegressionParametersParameterC()
        )

    name: Literal["LogisticRegression"]
    parameters: Parameters


@dataclass
class PLSRegression(Algorithm):
    """Cross decomposition using partial least squares (PLS)."""

    @type_name("PLSParams")
    @dataclass
    class Parameters:
        @dataclass
        class NComponents:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=5, metadata=schema(min=2))

        n_components: NComponents = NComponents()

    name: Literal["PLSRegression"]
    parameters: Parameters


@dataclass
class PLS(PLSRegression):
    name: Literal["PLS"]


class RandomForestMaxFeatures(str, Enum):
    AUTO = "auto"
    SQRT = "sqrt"
    LOG2 = "log2"


@dataclass
class RandomForestClassifier(Algorithm):
    """A random forest classifier.

    A random forest is a meta estimator
    that fits a number of decision tree classifiers
    on various sub-samples of the dataset
    and uses averaging
    to improve the predictive accuracy
    and control over-fitting.
    """

    @type_name("RandomForestClassifierParams")
    @dataclass
    class Parameters:
        @dataclass
        class RandomForestClassifierParametersMaxDepth:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=32, metadata=schema(min=1))

        @dataclass
        class RandomForestClassifierParametersNEstimators:
            low: int = field(default=10, metadata=schema(min=1))
            high: int = field(default=250, metadata=schema(min=1))

        max_depth: RandomForestClassifierParametersMaxDepth = (
            RandomForestClassifierParametersMaxDepth()
        )
        n_estimators: RandomForestClassifierParametersNEstimators = (
            RandomForestClassifierParametersNEstimators()
        )
        max_features: List[RandomForestMaxFeatures] = field(
            default_factory=lambda: [RandomForestMaxFeatures.AUTO]
        )

    name: Literal["RandomForestClassifier"]
    parameters: Parameters


@dataclass
class RandomForestRegressor(Algorithm):
    """A random forest regressor."""

    @type_name("RandomForestRegressorParams")
    @dataclass
    class Parameters:
        @dataclass
        class RandomForestRegressorParametersMaxDepth:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=32, metadata=schema(min=1))

        @dataclass
        class RandomForestRegressorParametersNEstimators:
            low: int = field(default=10, metadata=schema(min=1))
            high: int = field(default=250, metadata=schema(min=1))

        max_depth: RandomForestRegressorParametersMaxDepth = (
            RandomForestRegressorParametersMaxDepth()
        )
        n_estimators: RandomForestRegressorParametersNEstimators = (
            RandomForestRegressorParametersNEstimators()
        )
        max_features: List[RandomForestMaxFeatures] = field(
            default_factory=lambda: [RandomForestMaxFeatures.AUTO]
        )

    name: Literal["RandomForestRegressor"]
    parameters: Parameters


@dataclass
class RandomForest(Algorithm):
    """Deprecated."""

    @type_name("RandomForestParams")
    @dataclass
    class Parameters:
        @dataclass
        class RandomForestParametersMaxDepth:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=32, metadata=schema(min=1))

        @dataclass
        class RandomForestParametersNEstimators:
            low: int = field(default=10, metadata=schema(min=1))
            high: int = field(default=250, metadata=schema(min=1))

        max_depth: RandomForestParametersMaxDepth = RandomForestParametersMaxDepth()
        n_estimators: RandomForestParametersNEstimators = (
            RandomForestParametersNEstimators()
        )
        max_features: List[RandomForestMaxFeatures] = field(
            default_factory=lambda: [RandomForestMaxFeatures.AUTO]
        )

    name: Literal["RandomForest"]
    parameters: Parameters


@dataclass
class Ridge(Algorithm):
    """Linear least squares with l2 regularization.

    This model solves a regression model
    where the loss function is the linear least squares function
    and regularization is given by the l2-norm.
    Also known as Ridge Regression or Tikhonov regularization.
    """

    @type_name("RidgeParams")
    @dataclass
    class Parameters:
        @type_name("RidgeParamsAlpha")
        @dataclass
        class Alpha:
            low: float = field(default=0.0, metadata=schema(min=0.0))
            high: float = field(default=2.0, metadata=schema(min=0.0))

        alpha: Alpha = Alpha()

    name: Literal["Ridge"]
    parameters: Parameters


@dataclass
class SVC(Algorithm):
    """C-Support Vector Classification."""

    @type_name("SVCParams")
    @dataclass
    class Parameters:
        @dataclass
        class SVCParametersParameterC:
            low: float = field(default=1e-10, metadata=schema(min=1e-30))
            high: float = field(default=1e02, metadata=schema(max=1e10))

        @dataclass
        class Gamma:
            low: float = field(default=1e-4, metadata=schema(min=1e-9))
            high: float = field(default=1e2, metadata=schema(max=1e3))

        C: SVCParametersParameterC = SVCParametersParameterC()
        gamma: Gamma = Gamma()

    name: Literal["SVC"]
    parameters: Parameters


@dataclass
class SVR(Algorithm):
    """Epsilon-Support Vector Regression."""

    @type_name("SVRParams")
    @dataclass
    class Parameters:
        @dataclass
        class SVRParametersParameterC:
            low: float = field(default=1e-10, metadata=schema(min=1e-30))
            high: float = field(default=1e02, metadata=schema(max=1e10))

        @dataclass
        class SVRParametersGamma:
            low: float = field(default=1e-4, metadata=schema(min=1e-9))
            high: float = field(default=1e2, metadata=schema(max=1e3))

        C: SVRParametersParameterC = SVRParametersParameterC()
        gamma: SVRParametersGamma = SVRParametersGamma()

    name: Literal["SVR"]
    parameters: Parameters


@dataclass
class XGBRegressor(Algorithm):
    """XGBoost: gradient boosting trees algorithm."""

    @type_name("XGBregressorParams")
    @dataclass
    class Parameters:
        @dataclass
        class MaxDepth:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=32, metadata=schema(min=1))

        @dataclass
        class NEstimators:
            low: int = field(default=10, metadata=schema(min=1))
            high: int = field(default=250, metadata=schema(min=1))

        @dataclass
        class LearningRate:
            low: float = field(default=0.1, metadata=schema(min=0.0001))
            high: float = field(default=0.1, metadata=schema(min=0.001))

        max_depth: MaxDepth = MaxDepth()
        n_estimators: NEstimators = NEstimators()
        learning_rate: LearningRate = LearningRate()

    parameters: Parameters = Parameters()
    name: Literal["XGBRegressor"] = "XGBRegressor"


@dataclass
class XGBregressor(XGBRegressor):
    # Deprecated
    name: Literal["XGBregressor"] = "XGBregressor"


AnyRegressionAlgorithm = Union[
    Lasso,
    PLSRegression,
    PLS,  # Deprecated
    RandomForestRegressor,
    RandomForest,  # Deprecated
    Ridge,
    SVR,
    XGBRegressor,
    XGBregressor,  # Deprecated
]

AnyClassificationAlgorithm = Union[
    AdaBoostClassifier,
    LogisticRegression,
    RandomForestClassifier,
    RandomForest,  # Deprecated
    SVC,
]

AnyAlgorithm = Union[AnyRegressionAlgorithm, AnyClassificationAlgorithm]


def replace_rf(alg: AnyAlgorithm, mode: ModelMode) -> AnyAlgorithm:
    if isinstance(alg, RandomForest):
        if mode == ModelMode.REGRESSION:
            return RandomForestRegressor.new(**mkdict(alg.parameters))
        elif mode == ModelMode.CLASSIFICATION:
            return RandomForestClassifier.new(**mkdict(alg.parameters))
        else:
            raise ValueError(f"Unexpected mode: {mode}")
    else:
        return alg


def isanyof(obj: Any, classes: Iterable[Type]) -> bool:
    return any(isinstance(obj, cls) for cls in classes)


def detect_mode_from_algs(algs: List[AnyAlgorithm]) -> ModelMode:

    # Getting algs could be replaced by
    # typing.get_args(AnyRegressionAlgorithm) from Python 3.8.
    regression_algs = [
        Lasso,
        PLSRegression,
        RandomForestRegressor,
        Ridge,
        SVR,
        XGBRegressor,
    ]
    classification_algs = [
        AdaBoostClassifier,
        LogisticRegression,
        RandomForestClassifier,
        SVC,
    ]
    if all(isanyof(alg, regression_algs) for alg in algs):
        mode = ModelMode.REGRESSION
    elif all(isanyof(alg, classification_algs) for alg in algs):
        mode = ModelMode.CLASSIFICATION
    else:
        raise ValueError(
            f"Provided algorithms ({algs}) "
            f"are neither only regression ({regression_algs}),",
            f"nor only classification ({classification_algs})",
        )
    return mode


def copy_path_for_scaled_descriptor(
    descriptors: List[MolDescriptor], dataset: Dataset
) -> None:
    def recursive_copy_path_for_scaled_descriptor(d: MolDescriptor) -> None:
        if isinstance(d, ScaledDescriptor):
            scaler = d.parameters.scaler
            if (
                isinstance(scaler, UnfittedSklearnScaler)
                and scaler.mol_data.file_path is None
            ):
                d.set_unfitted_scaler_data(
                    dataset.training_dataset_file, dataset.input_column
                )
        elif isinstance(d, CompositeDescriptor):
            for dd in d.parameters.descriptors:
                recursive_copy_path_for_scaled_descriptor(dd)

    for d in descriptors:
        recursive_copy_path_for_scaled_descriptor(d)


@dataclass
class OptimizationConfig:
    """Optimization configuration.

    This is configuration for hyperparameter optimization.
    It roughly corresponds to Optuna Study.
    """

    @dataclass
    class Settings:
        """Optimization settings."""

        mode: Optional[ModelMode] = field(
            default=None, metadata=schema(title="Classification or regression.")
        )

        cross_validation: int = field(
            default=5,
            metadata=schema(
                min=1,
                title="Number of folds in cross-validation, use '1' to disable cross-validation",
            ),
        )

        shuffle: bool = field(
            default=False,
            metadata=schema(
                title="Whether or not to shuffle the data for cross-validation"
            ),
        )

        direction: Optional[OptimizationDirection] = field(
            default=None, metadata=schema(title="Maximization or minimization")
        )

        scoring: Union[RegressionScore, ClassificationScore, str] = field(
            default=None, metadata=schema(title="Scoring metric")
        )

        n_trials: Optional[int] = field(
            default=300,
            metadata=schema(
                min=0,  # Zero for no optimization (but run preprocessing).
                title="Total number of trials",
                description="Initial guess? About 300. Check optimization progress to see if optimization converged.",
            ),
        )

        n_jobs: Optional[int] = field(
            default=-1,
            metadata=schema(
                title="Number of parallel jobs, set to '-1' to use as many jobs as CPU cores available"
            ),
        )

        n_startup_trials: int = field(
            default=50,
            metadata=schema(
                title="Number of initial (startup, exploratory) random trials",
                description="Take this number of trials out of total number of trials "
                "and do random exploratory search, "
                "without performing selection/optimization. "
                "Use to not get stuck early on in a local minimum.",
            ),
        )

        random_seed: Optional[int] = field(
            default=None,
            metadata=schema(
                title="Seed for random number generator",
                description="Seed for random number generator."
                " Set to an integer value to get reproducible results."
                " Set to None to initialize at random.",
            ),
        )

        optuna_storage: Optional[str] = field(
            default=None,
            metadata=schema(
                title="Database URL for Optuna Storage",
                description="Database URL for Optuna Storage."
                " Set to None/null to use default in-memory storage."
                " Example: 'sqlite:///optuna_storage.db' for file-based SQLite3 storage.",
            ),
        )

        track_to_mlflow: Optional[bool] = field(
            default=True,
            metadata=schema(
                title="Track to MLFlow",
                description="Set to True to use MLFlow tracking UI,"
                " set to False to disable MLFlow.",
            ),
        )

        tracking_rest_endpoint: Optional[str] = field(
            default=None,
            metadata=schema(
                title="URL to track Optuna progress using internal format"
            )
        )

    name: str = field(
        default="",
        metadata=schema(title="Name", description="Name of the optimization job."),
    )
    description: str = field(
        default="",
        metadata=schema(
            title="Description", description="Description of the optimization job."
        ),
    )
    data: Dataset = field(default=None, metadata=schema(title="Dataset"))
    mode: Optional[ModelMode] = field(
        default=None, metadata=schema(title="Classification or regression")
    )  # For GUI compatibility.
    algorithms: List[AnyAlgorithm] = None
    descriptors: List[MolDescriptor] = None
    settings: Settings = None
    visualization: Optional[Visualization] = field(default=None)
    task: Literal["optimization"] = "optimization"

    def __post_init__(self):
        if self.mode is None and self.settings.mode is not None:
            self.mode = self.settings.mode
        elif self.settings.mode is None and self.mode is not None:
            self.settings.mode = self.mode
        elif (
            self.settings.mode is not None
            and self.mode is not None
            and self.settings.mode != self.mode
        ):
            raise ValueError(
                f"Value mismatch: mode={self.mode} settings.mode={self.settings.mode}"
            )
        elif self.settings.mode is None and self.mode is None:
            mode = detect_mode_from_algs(self.algorithms)
            self.mode = mode
            self.settings.mode = mode

        self.algorithms = [
            replace_rf(alg, self.settings.mode) for alg in self.algorithms
        ]

        copy_path_for_scaled_descriptor(self.descriptors, self.data)

        if self.settings.scoring is None:
            if self.settings.mode == ModelMode.REGRESSION:
                self.settings.scoring = "r2"
            elif self.settings.mode == ModelMode.CLASSIFICATION:
                self.settings.scoring = "roc_auc"
        elif isinstance(self.settings.scoring, Enum):
            self.settings.scoring = self.settings.scoring.value
