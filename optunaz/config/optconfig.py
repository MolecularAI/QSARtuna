import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional, List, Iterable, Type, Any, Literal, Annotated

from apischema import schema, type_name, serialize
from apischema.metadata import none_as_undefined, required

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
    ScalingFittingError,
)
from optunaz.utils import md5_hash
from optunaz.utils.preprocessing.splitter import (
    AnyCvSplitter,
    Random,
)

from joblib import Memory
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)


class ClassificationScore(str, Enum):
    ACCURACY = "accuracy"
    AVERAGE_PRECISION = "average_precision"
    AUC_PR_CAL = "auc_pr_cal"
    BALANCED_ACCURACY = "balanced_accuracy"
    BEDROC = "bedroc_score"
    CONCORDANCE_INDEX = "concordance_index"
    F1 = "f1"
    F1_MACRO = "f1_macro"
    F1_MICRO = "f1_micro"
    F1_WEIGHTED = "f1_weighted"
    JACCARD = "jaccard"
    JACCARD_MACRO = "jaccard_macro"
    JACCARD_MICRO = "jaccard_micro"
    JACCARD_WEIGHTED = "jaccard_weighted"
    NEG_BRIER_SCORE = "neg_brier_score"
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
    """AdaBoost Classifier.

    An AdaBoost classifier is a meta-estimator
    that begins by fitting a classifier on the original dataset
    and then fits additional copies of the classifier on the same dataset
    but where the weights of incorrectly classified instances are adjusted
    such that subsequent classifiers focus more on difficult cases.
    """

    @type_name("AdaBoostClassifierParams")
    @dataclass
    class Parameters:
        @dataclass
        class AdaBoostClassifierParametersNEstimators:
            low: int = field(default=3, metadata=schema(title="low", min=1))
            high: int = field(default=100, metadata=schema(title="high", min=1))

        @dataclass
        class AdaBoostClassifierParametersLearningRate:
            low: float = field(default=1.0, metadata=schema(title="low", min=0.0001))
            high: float = field(default=1.0, metadata=schema(title="high", min=0.001))

        n_estimators: Annotated[
            AdaBoostClassifierParametersNEstimators,
            schema(
                title="n_estimators",
                description="The maximum number of estimators"
                " at which boosting is terminated."
                " In case of perfect fit, the learning procedure is stopped early.",
            ),
        ] = AdaBoostClassifierParametersNEstimators()

        learning_rate: Annotated[
            AdaBoostClassifierParametersLearningRate,
            schema(
                title="learning_rate",
                description="Weight applied to each classifier"
                "at each boosting iteration. A higher learning rate"
                "increases the contribution of each classifier. "
                "There is a trade-off between the learning_rate"
                "and n_estimators parameters.",
            ),
        ] = AdaBoostClassifierParametersLearningRate()

    name: Literal["AdaBoostClassifier"]
    parameters: Parameters


@dataclass
class Lasso(Algorithm):
    """Lasso regression.

    Lasso is a Linear Model trained with L1 prior as regularizer.

    The Lasso is a linear model that estimates sparse coefficients.
    It tends to prefer solutions with fewer non-zero coefficients,
    effectively reducing the number of features
    upon which the given solution is dependent.
    """

    @type_name("LassoParams")
    @dataclass
    class Parameters:
        @dataclass
        class LassoParametersAlpha:
            low: float = field(default=0.0, metadata=schema(title="low", min=0))
            high: float = field(default=2.0, metadata=schema(title="high", min=0))

        alpha: Annotated[
            LassoParametersAlpha,
            schema(
                title="Alpha",
                description="Constant that multiplies the L1 term,"
                " controlling regularization strength."
                " alpha must be a non-negative float i.e. in [0, inf)."
                " When alpha = 0, the objective is equivalent to ordinary least squares,"
                " solved by the LinearRegression object."
                " For numerical reasons,"
                " using alpha = 0 with the Lasso object is not advised."
                " Instead, you should use the LinearRegression object.",
            ),
        ] = LassoParametersAlpha()

    name: Literal["Lasso"]
    parameters: Parameters


class KNeighborsWeights(str, Enum):
    """Method used to define the weights for a K-Neighbors Classifier"""

    UNIFORM = "uniform"
    "uniform weights. All points in each neighborhood are weighted equally."
    DISTANCE = "distance"
    """weight points by the inverse of their distance so closer neighbors for a query will have greater \
     influence than further neighbors"""


class KNeighborsMetric(str, Enum):
    """Metric used to define the weights for a K-Neighbors Classifier"""

    MINKOWSKI = "minkowski"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


@dataclass
class KNeighborsClassifier(Algorithm):
    """KNeighborsClassifier.

    Classifier implementing the k-nearest neighbors vote.

    The principle behind nearest neighbor methods is to find a predefined number of training samples closest in
    distance to the new point, and predict the label from these. The number of samples is a user-defined constant
    for k-nearest neighbor learning. Despite its simplicity, nearest neighbors is successful in a large number of
    classification problems
    """

    @type_name("KNeighborsClassifierParams")
    @dataclass
    class Parameters:
        @dataclass
        class KNeighborsClassifierParametersN_Neighbors:
            low: float = field(default=1, metadata=schema(title="low", min=0))
            high: float = field(default=10, metadata=schema(title="high", min=0))

        n_neighbors: Annotated[
            KNeighborsClassifierParametersN_Neighbors,
            schema(
                title="N Neighbors",
                description="Number of neighbors to use by default for kneighbors queries.",
            ),
        ] = KNeighborsClassifierParametersN_Neighbors()
        weights: Annotated[
            List[KNeighborsWeights],
            schema(
                title="Weights",
                description="Weight function used in prediction",
            ),
        ] = field(default_factory=lambda: [KNeighborsWeights.UNIFORM])
        metric: Annotated[
            List[KNeighborsMetric],
            schema(
                title="Metric",
                description="Metric to use for distance computation."
                "The default of “minkowski” results in the standard Euclidean distance",
            ),
        ] = field(default_factory=lambda: [KNeighborsMetric.MINKOWSKI])

    name: Literal["KNeighborsClassifier"]
    parameters: Parameters


@dataclass
class KNeighborsRegressor(Algorithm):
    """KNeighborsRegressor.

    Regressor implementing the k-nearest neighbors vote.

    The principle behind nearest neighbor methods is to find a predefined number of training samples closest in
    distance to the new point, and predict the label from these. The number of samples is a user-defined constant
    for k-nearest neighbor learning. Despite its simplicity, nearest neighbors is successful in a large number of
    classification problems
    """

    @type_name("KNeighborsRegressorParams")
    @dataclass
    class Parameters:
        @dataclass
        class KNeighborsRegressorParametersN_Neighbors:
            low: float = field(default=1, metadata=schema(title="low", min=0))
            high: float = field(default=10, metadata=schema(title="high", min=0))

        n_neighbors: Annotated[
            KNeighborsRegressorParametersN_Neighbors,
            schema(
                title="N Neighbors",
                description="Number of neighbors to use by default for kneighbors queries.",
            ),
        ] = KNeighborsRegressorParametersN_Neighbors()
        weights: Annotated[
            List[KNeighborsWeights],
            schema(
                title="Weights",
                description="Weight function used in prediction",
            ),
        ] = field(default_factory=lambda: [KNeighborsWeights.UNIFORM])
        metric: Annotated[
            List[KNeighborsMetric],
            schema(
                title="Metric",
                description="Metric to use for distance computation."
                "The default of “minkowski” results in the standard Euclidean distance",
            ),
        ] = field(default_factory=lambda: [KNeighborsMetric.MINKOWSKI])

    name: Literal["KNeighborsRegressor"]
    parameters: Parameters


@dataclass
class LogisticRegression(Algorithm):
    """Logistic Regression classifier.

    Logistic regression, despite its name,
    is a linear model for classification rather than regression.
    Logistic regression is also known in the literature as logit regression,
    maximum-entropy classification (MaxEnt) or the log-linear classifier.
    In this model,
    the probabilities describing the possible outcomes of a single trial
    are modeled using a logistic function.
    """

    @type_name("LogisticRegressionParams")
    @dataclass
    class Parameters:
        @dataclass
        class LogisticRegressionParametersParameterC:
            low: float = field(default=1.0, metadata=schema(title="low", min=0.001))
            high: float = field(default=1.0, metadata=schema(title="high", max=1000))

        solver: Annotated[
            List[str],
            schema(
                title="Solver",
                description="List of solvers to try."
                " Note ‘sag’ and ‘saga’ fast convergence"
                " is only guaranteed on features with approximately the same scale."
                " You can preprocess the data with a scaler.",
            ),
        ] = field(default_factory=lambda: ["newton-cg", "lbfgs", "sag", "saga"])

        C: Annotated[
            LogisticRegressionParametersParameterC,
            schema(
                title="C",
                description="Inverse of regularization strength;"
                " must be a positive float."
                " Like in support vector machines,"
                " smaller values specify stronger regularization.",
            ),
        ] = LogisticRegressionParametersParameterC()

    name: Literal["LogisticRegression"]
    parameters: Parameters


@dataclass
class PLSRegression(Algorithm):
    """PLS regression (Cross decomposition using partial least squares).

    PLS is a form of regularized linear regression
    where the number of components controls the strength of the regularization.

    Cross decomposition algorithms
    find the fundamental relations between two matrices (X and Y).
    They are latent variable approaches
    to modeling the covariance structures in these two spaces.
    They will try to find the multidimensional direction in the X space
    that explains the maximum multidimensional variance direction in the Y space.
    In other words,
    PLS projects both X and Y into a lower-dimensional subspace
    such that the covariance between transformed(X) and transformed(Y) is maximal.
    """

    @type_name("PLSParams")
    @dataclass
    class Parameters:
        @dataclass
        class NComponents:
            low: int = field(default=2, metadata=schema(title="low", min=1))
            high: int = field(default=5, metadata=schema(title="high", min=2))

        n_components: Annotated[
            NComponents,
            schema(
                title="n_components",
                description="Number of components to keep."
                " Should be in [1, min(n_samples, n_features, n_targets)].",
            ),
        ] = NComponents()

    name: Literal["PLSRegression"]
    parameters: Parameters


class RandomForestMaxFeatures(str, Enum):
    """Method used to define the maximum number of features in a Random Forest"""

    AUTO = "auto"
    "Auto sets `max_features=sqrt(n_features)`."
    SQRT = "sqrt"
    "Square root sets `max_features=sqrt(n_features)`."
    LOG2 = "log2"
    "Log2 sets `max_features=log2(n_features)`."


@dataclass
class RandomForestClassifier(Algorithm):
    """Random Forest classifier.

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
            low: int = field(default=2, metadata=schema(title="low", min=1))
            high: int = field(default=32, metadata=schema(title="high", min=1))

        @dataclass
        class RandomForestClassifierParametersNEstimators:
            low: int = field(default=10, metadata=schema(title="low", min=1))
            high: int = field(default=250, metadata=schema(title="high", min=1))

        max_depth: Annotated[
            RandomForestClassifierParametersMaxDepth,
            schema(
                title="max_depth",
                description="The maximum depth of the tree.",
            ),
        ] = RandomForestClassifierParametersMaxDepth()

        n_estimators: Annotated[
            RandomForestClassifierParametersNEstimators,
            schema(
                title="n_estimators",
                description="The number of trees in the forest.",
            ),
        ] = RandomForestClassifierParametersNEstimators()

        max_features: Annotated[
            List[RandomForestMaxFeatures],
            schema(
                title="max_features",
                description="The number of features to consider"
                "when looking for the best split:  If auto, then"
                "consider max_features features at each split."
                " - If “auto”, then `max_features=n_features`."
                " - If “sqrt”, then `max_features=sqrt(n_features)`."
                " - If “log2”, then `max_features=log2(n_features)`.",
            ),
        ] = field(default_factory=lambda: [RandomForestMaxFeatures.AUTO])

    name: Literal["RandomForestClassifier"]
    parameters: Parameters


@dataclass
class RandomForestRegressor(Algorithm):
    """Random Forest regression.

    A random forest is a meta estimator
    that fits a number of classifying decision trees
    on various sub-samples of the dataset
    and uses averaging
    to improve the predictive accuracy
    and control over-fitting.
    """

    @type_name("RandomForestRegressorParams")
    @dataclass
    class Parameters:
        @dataclass
        class RandomForestRegressorParametersMaxDepth:
            low: int = field(default=2, metadata=schema(title="low", min=1))
            high: int = field(default=32, metadata=schema(title="high", min=1))

        @dataclass
        class RandomForestRegressorParametersNEstimators:
            low: int = field(default=10, metadata=schema(title="low", min=1))
            high: int = field(default=250, metadata=schema(title="high", min=1))

        max_depth: Annotated[
            RandomForestRegressorParametersMaxDepth,
            schema(
                title="max_depth",
                description="The maximum depth of the tree.",
            ),
        ] = RandomForestRegressorParametersMaxDepth()

        n_estimators: Annotated[
            RandomForestRegressorParametersNEstimators,
            schema(
                title="n_estimators",
                description="The number of trees in the forest.",
            ),
        ] = RandomForestRegressorParametersNEstimators()

        max_features: Annotated[
            List[RandomForestMaxFeatures],
            schema(
                title="max_features",
                description="The number of features to consider"
                "when looking for the best split:  If auto, then"
                "consider max_features features at each split."
                " - If “auto”, then `max_features=n_features`."
                " - If “sqrt”, then `max_features=sqrt(n_features)`."
                " - If “log2”, then `max_features=log2(n_features)`.",
            ),
        ] = field(default_factory=lambda: [RandomForestMaxFeatures.AUTO])

    name: Literal["RandomForestRegressor"]
    parameters: Parameters


@dataclass
class Ridge(Algorithm):
    """Ridge Regression (Linear least squares with l2 regularization).

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
            low: float = field(default=0.0, metadata=schema(title="low", min=0.0))
            high: float = field(default=2.0, metadata=schema(title="high", min=0.0))

        alpha: Annotated[
            Alpha,
            schema(
                title="alpha",
                description="Constant that multiplies the L2 term, controlling regularization strength",
            ),
        ] = Alpha()

    name: Literal["Ridge"]
    parameters: Parameters


@dataclass
class SVC(Algorithm):
    """SVC classifier (C-Support Vector Classification).

    The implementation is based on libsvm.
    The fit time scales at least quadratically with the number of samples
    and may be impractical beyond tens of thousands of samples.
    """

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

        C: Annotated[
            SVCParametersParameterC,
            schema(
                title="C",
                description="Regularization parameter."
                " The strength of the regularization is inversely proportional to C."
                " Must be strictly positive."
                " The penalty is a squared l2 penalty.",
            ),
        ] = SVCParametersParameterC()

        gamma: Annotated[
            Gamma,
            schema(title="gamma", description="Kernel coefficient"),
        ] = Gamma()

    name: Literal["SVC"]
    parameters: Parameters


@dataclass
class SVR(Algorithm):
    """SVR regression (Epsilon-Support Vector Regression).

    The implementation is based on libsvm.
    The fit time complexity is more than quadratic with the number of samples
    which makes it hard to scale to datasets with more than a couple of 10000 samples.
    """

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

        C: Annotated[
            SVRParametersParameterC,
            schema(
                title="C",
                description="Regularization parameter."
                " The strength of the regularization is inversely proportional to C."
                " Must be strictly positive."
                " The penalty is a squared l2 penalty.",
            ),
        ] = SVRParametersParameterC()

        gamma: Annotated[
            SVRParametersGamma,
            schema(title="gamma", description="Kernel coefficient"),
        ] = SVRParametersGamma()

    name: Literal["SVR"]
    parameters: Parameters


@dataclass
class XGBRegressor(Algorithm):
    """XGBoost regression (gradient boosting trees algorithm).

    XGBoost stands for “Extreme Gradient Boosting”,
    where the term “Gradient Boosting” originates from the paper
    Greedy Function Approximation: A Gradient Boosting Machine, by Friedman.
    """

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

        max_depth: Annotated[
            MaxDepth,
            schema(
                title="max_depth",
                description="Maximum tree depth for base learners.",
            ),
        ] = MaxDepth()

        n_estimators: Annotated[
            NEstimators,
            schema(
                title="n_estimators",
                description="Number of gradient boosted trees."
                "Equivalent to number of boosting rounds.",
            ),
        ] = NEstimators()

        learning_rate: Annotated[
            LearningRate,
            schema(
                title="learning_rate",
                description="Weight applied to each classifier"
                "at each boosting iteration. A higher learning rate"
                "increases the contribution of each classifier. "
                "There is a trade-off between the learning_rate"
                "and n_estimators parameters.",
            ),
        ] = LearningRate()

    parameters: Parameters = Parameters()
    name: Literal["XGBRegressor"] = "XGBRegressor"


class PRFClassifierMaxFeatures(str, Enum):
    """Method used to define the maximum number of features in a Probabilistic Random Forest"""

    AUTO = "auto"
    "Auto sets `max_features=sqrt(n_features)`."
    SQRT = "sqrt"
    "Square root sets `max_features=sqrt(n_features)`."
    LOG2 = "log2"
    "Log2 sets `max_features=log2(n_features)`."


@dataclass
class PRFClassifier(Algorithm):
    """PRF (Probabilistic Random Forest).

    PRF can be seen as a hybrid between regression and classification algorithms.
    Similar to regression algorithms,
    PRF takes as input real-valued probabilities,
    usually from Probabilistic Threshold Representation (PTR).
    However, similar to classification algorithms,
    it predicts probability of belonging to active or inactive class.
    """

    @type_name("PRFClassifierParams")
    @dataclass
    class Parameters:
        @dataclass
        class PRFClassifierParametersNEstimators:
            low: int = field(default=10, metadata=schema(min=1))
            high: int = field(default=250, metadata=schema(min=1))

        @dataclass
        class PRFClassifierParametersMaxDepth:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=32, metadata=schema(min=1))

        @dataclass
        class PRFClassifierParametersMinPySumLeaf:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=5, metadata=schema(min=1))

        use_py_gini: Annotated[
            int,
            schema(
                min=0,
                max=1,
                title="Use pY GINI",
                description="The probability of y is used in GINI when this is True",
            ),
        ] = field(default=1)

        use_py_leafs: Annotated[
            int,
            schema(
                min=0,
                max=1,
                title="Use pY leafs",
                description="The probability of y is used in leaves when this is True",
            ),
        ] = field(default=1)

        max_depth: Annotated[
            PRFClassifierParametersMaxDepth,
            schema(
                title="max_depth",
                description="The maximum depth of the tree.",
            ),
        ] = PRFClassifierParametersMaxDepth()

        n_estimators: Annotated[
            PRFClassifierParametersNEstimators,
            schema(
                title="n_estimators",
                description="The number of trees in the forest.",
            ),
        ] = PRFClassifierParametersNEstimators()

        max_features: Annotated[
            List[PRFClassifierMaxFeatures],
            schema(
                title="max_features",
                description="The number of features to consider"
                "when looking for the best split: "
                " - If “auto”, then `max_features=sqrt(n_features)`."
                " - If “sqrt”, then `max_features=sqrt(n_features)`."
                " - If “log2”, then `max_features=log2(n_features)`.",
            ),
        ] = field(default_factory=lambda: [PRFClassifierMaxFeatures.AUTO])

        min_py_sum_leaf: Annotated[
            PRFClassifierParametersMinPySumLeaf,
            schema(
                title="n_estimators",
                description="This parameter allows tree pruning when the propagation "
                "probability is small, thus reducing computation time. "
                "This value defines the probability threshold, `pth` as"
                " described in the Selective propagation scheme in the "
                "original publication `Probabilistic Random Forest: A "
                "machine learning algorithm for noisy datasets`",
            ),
        ] = PRFClassifierParametersMinPySumLeaf()

    name: Literal["PRFClassifier"]
    parameters: Parameters


class ChemPropActivation(str, Enum):
    """
    The activation function to use within the network.
    See https://chemprop.readthedocs.io/en/latest/args.html#chemprop.args.TrainArgs.activation for details
    """

    RELU = "ReLU"
    TANH = "tanh"
    LEAKYRELU = "LeakyReLU"
    PRELU = "PReLU"
    SELU = "SELU"
    ELU = "ELU"


class ChemPropFeatures_Generator(str, Enum):
    """
    Features generators are used for computing additional molecule-level features that are appended
    after message passing.
    See https://chemprop.readthedocs.io/en/latest/features.html#features-generators for details.
    """

    NONE = "none"
    "Turns off the features generator function."
    MORGAN = "morgan"
    "Generates a binary Morgan fingerprint for a molecule."
    MORGAN_COUNT = "morgan_count"
    "Generates a counts-based Morgan fingerprint for a molecule."
    RDKIT_2D = "rdkit_2d"
    "Generates RDKit 2D features for a molecule."
    RDKIT_2D_NORMALIZED = "rdkit_2d_normalized"
    "Generates RDKit 2D normalized features for a molecule."


class ChemPropAggregation(str, Enum):
    """Atom-level representations from the MPNN"""

    MEAN = "mean"
    "Representations averaged over all atoms of a molecule"
    SUM = "sum"
    "Representations summed over all atoms of a molecule"
    NORM = "norm"
    "Representations summed up and divided by a constant (default=100)"


@dataclass
class ChemPropClassifier(Algorithm):
    """Chemprop Classifier without hyperopt

    Chemprop is an open-source package for training deep learning models for molecular property prediction. ChemProp
    trains two networks; a Directed Message Passing Neural Network (D-MPNN) to encode a graph representation of
    molecules, and a Feed Forward Neural Network (FFNN); a standard multi-layer perceptron trained to predict the
    target property using D-MPNN encoding. It was first presented in the paper "Analyzing Learned Molecular
    Representations for Property Prediction".  This implementation will use Optuna to optimse parameters instead of
    Hyperopt (as in the original implementation of ChemProp).
    """

    @type_name("ChemPropClassifierParams")
    @dataclass
    class Parameters:
        @type_name("ChemPropClassifierAggregation_Norm")
        @dataclass
        class ChemPropParametersAggregation_Norm:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=200, metadata=schema(min=1))
            q: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropClassifierBatch_Size")
        @dataclass
        class ChemPropParametersBatch_Size:
            low: int = field(default=5, metadata=schema(min=5))
            high: int = field(default=200, metadata=schema(min=5))
            q: int = field(default=5, metadata=schema(min=5))

        @type_name("ChemPropClassifierDepth")
        @dataclass
        class ChemPropParametersDepth:
            low: int = field(default=2, metadata=schema(min=2))
            high: int = field(default=6, metadata=schema(min=2))
            q: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropClassifierDropout")
        @dataclass
        class ChemPropParametersDropout:
            low: float = field(default=0.0, metadata=schema(min=0, max=1))
            high: float = field(default=0.4, metadata=schema(min=0, max=1))
            q: float = field(default=0.04, metadata=schema(min=0.04))

        @type_name("ChemPropClassifierFFN_Hidden_Size")
        @dataclass
        class ChemPropParametersFFN_Hidden_Size:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            q: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropClassifierFFN_Num_Layers")
        @dataclass
        class ChemPropParametersFFN_Num_Layers:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=3, metadata=schema(min=1))
            q: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropClassifierFinal_Lr_Ratio_Exp")
        @dataclass
        class ChemPropParametersFinal_Lr_Ratio_Exp:
            low: int = field(default=-4, metadata=schema(min=-4))
            high: int = field(default=0, metadata=schema(min=-4))

        @type_name("ChemPropClassifierHidden_Size")
        @dataclass
        class ChemPropParametersHidden_Size:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            q: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropClassifierInit_Lr_Ratio_Exp")
        @dataclass
        class ChemPropParametersInit_Lr_Ratio_Exp:
            low: int = field(default=-4, metadata=schema(min=-4))
            high: int = field(default=0, metadata=schema(min=-4))

        @type_name("ChemPropClassifierMax_Lr_Exp")
        @dataclass
        class ChemPropParametersMax_Lr_Exp:
            low: int = field(default=-6, metadata=schema(min=-6))
            high: int = field(default=-2, metadata=schema(min=-6))

        @type_name("ChemPropClassifierWarmup_Epochs_Ratio")
        @dataclass
        class ChemPropParametersWarmup_Epochs_Ratio:
            low: float = field(default=0.1, metadata=schema(min=0.1))
            high: float = field(default=0.1, metadata=schema(min=0.1))
            q: float = field(default=0.1, metadata=schema(min=0.1))

        ensemble_size: Annotated[
            int,
            schema(
                min=1,
                max=5,
                title="Ensemble size",
                description="Number of ensembles with different weight initialisation (provides uncertainty)",
            ),
        ] = field(default=1)

        epochs: Annotated[
            int,
            schema(
                min=4,
                max=400,
                title="Epochs",
                description="Number of epochs to run (increasing this will increase run time)",
            ),
        ] = field(default=30)

        activation: Annotated[
            List[ChemPropActivation],
            schema(
                title="activation",
                description="Activation function applied to the "
                "output of the weighted sum of inputs",
            ),
        ] = field(default_factory=lambda: [ChemPropActivation.RELU])

        aggregation: Annotated[
            List[ChemPropAggregation],
            schema(
                title="aggregation",
                description="Aggregation scheme for atomic vectors into molecular vectors.",
            ),
        ] = field(default_factory=lambda: [ChemPropAggregation.MEAN])

        aggregation_norm: Annotated[
            ChemPropParametersAggregation_Norm,
            schema(
                title="aggregation_norm",
                description="For norm aggregation, number by which to divide summed up atomic features.",
            ),
        ] = ChemPropParametersAggregation_Norm()

        batch_size: Annotated[
            ChemPropParametersBatch_Size,
            schema(
                title="batch_size",
                description="How many samples per batch to load.",
            ),
        ] = ChemPropParametersBatch_Size()

        depth: Annotated[
            ChemPropParametersDepth,
            schema(
                title="depth",
                description="Number of message passing steps"
                "(distance of neighboring atoms visible when modelling).",
            ),
        ] = ChemPropParametersDepth()

        dropout: Annotated[
            ChemPropParametersDropout,
            schema(
                title="dropout",
                description="Dropout probability. During training, randomly zeroes"
                " some of the elements of the input tensor with probability `p`"
                " using samples from a Bernoulli distribution. Each channel will"
                " be zeroed out independently on every forward call.  This has"
                " proven to be an effective technique for regularization and "
                "preventing the co-adaptation of neurons",
            ),
        ] = ChemPropParametersDropout()

        features_generator: Annotated[
            List[ChemPropFeatures_Generator],
            schema(
                title="features_generator",
                description="Method of generating additional features.",
            ),
        ] = field(default_factory=lambda: [ChemPropFeatures_Generator.NONE])

        ffn_hidden_size: Annotated[
            ChemPropParametersFFN_Hidden_Size,
            schema(
                title="ffn_hidden_size",
                description="Dimensionality of hidden layers in the FFN.",
            ),
        ] = ChemPropParametersFFN_Hidden_Size()

        ffn_num_layers: Annotated[
            ChemPropParametersFFN_Num_Layers,
            schema(
                title="ffn_num_layers",
                description="Number of layers in the FFN after D-MPNN encoding.",
            ),
        ] = ChemPropParametersFFN_Num_Layers()

        final_lr_ratio_exp: Annotated[
            ChemPropParametersFinal_Lr_Ratio_Exp,
            schema(
                title="final_lr_ratio_exp",
                description="The exponential for the final learning rate.",
            ),
        ] = ChemPropParametersFinal_Lr_Ratio_Exp()

        hidden_size: Annotated[
            ChemPropParametersHidden_Size,
            schema(
                title="hidden_size",
                description="Size of the hidden bond message vectors in the D-MPNN",
            ),
        ] = ChemPropParametersHidden_Size()

        init_lr_ratio_exp: Annotated[
            ChemPropParametersInit_Lr_Ratio_Exp,
            schema(
                title="init_lr_ratio_exp",
                description="The exponential for the learning rate ratio.",
            ),
        ] = ChemPropParametersInit_Lr_Ratio_Exp()

        max_lr_exp: Annotated[
            ChemPropParametersMax_Lr_Exp,
            schema(
                title="max_lr_exp",
                description="The exponential for the maximum learning rate.",
            ),
        ] = ChemPropParametersMax_Lr_Exp()

        warmup_epochs_ratio: Annotated[
            ChemPropParametersWarmup_Epochs_Ratio,
            schema(
                title="warmup_epochs_ratio",
                description="Ratio for the number of epochs during which "
                "learning rate increases linearly from init_lr to max_lr."
                " Afterwards, learning rate decreases exponentially "
                "from max_lr to final_lr.",
            ),
        ] = ChemPropParametersWarmup_Epochs_Ratio()

    name: Literal["ChemPropClassifier"]
    parameters: Parameters


@dataclass
class ChemPropRegressor(Algorithm):
    """Chemprop Regressor

    Chemprop is an open-source package for training deep learning models for molecular property prediction. ChemProp
    trains two networks; a Directed Message Passing Neural Network (D-MPNN) to encode a graph representation of
    molecules, and a Feed Forward Neural Network (FFNN); a standard multi-layer perceptron trained to predict the
    target property using D-MPNN encoding. It was first presented in the paper "Analyzing Learned Molecular
    Representations for Property Prediction".  This implementation will use Optuna to optimse parameters instead of
    Hyperopt (as in the original implementation of ChemProp).
    """

    @type_name("ChemPropRegressorParams")
    @dataclass
    class Parameters:
        @type_name("ChemPropRegressorAggregation_Norm")
        @dataclass
        class ChemPropParametersAggregation_Norm:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=200, metadata=schema(min=1))
            q: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropRegressorBatch_Size")
        @dataclass
        class ChemPropParametersBatch_Size:
            low: int = field(default=5, metadata=schema(min=5))
            high: int = field(default=200, metadata=schema(min=5))
            q: int = field(default=5, metadata=schema(min=5))

        @type_name("ChemPropRegressorDepth")
        @dataclass
        class ChemPropParametersDepth:
            low: int = field(default=2, metadata=schema(min=2))
            high: int = field(default=6, metadata=schema(min=2))
            q: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropRegressorDropout")
        @dataclass
        class ChemPropParametersDropout:
            low: float = field(default=0.0, metadata=schema(min=0, max=1))
            high: float = field(default=0.4, metadata=schema(min=0, max=1))
            q: float = field(default=0.04, metadata=schema(min=0.04))

        @type_name("ChemPropRegressorFFN_Hidden_Size")
        @dataclass
        class ChemPropParametersFFN_Hidden_Size:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            q: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropRegressorFFN_Num_Layers")
        @dataclass
        class ChemPropParametersFFN_Num_Layers:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=3, metadata=schema(min=1))
            q: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropRegressorFinal_Lr_Ratio_Exp")
        @dataclass
        class ChemPropParametersFinal_Lr_Ratio_Exp:
            low: int = field(default=-4, metadata=schema(min=-4))
            high: int = field(default=0, metadata=schema(min=-4))

        @type_name("ChemPropRegressorHidden_Size")
        @dataclass
        class ChemPropParametersHidden_Size:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            q: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropRegressorInit_Lr_Ratio_Exp")
        @dataclass
        class ChemPropParametersInit_Lr_Ratio_Exp:
            low: int = field(default=-4, metadata=schema(min=-4))
            high: int = field(default=0, metadata=schema(min=-4))

        @type_name("ChemPropRegressorMax_Lr_Exp")
        @dataclass
        class ChemPropParametersMax_Lr_Exp:
            low: int = field(default=-6, metadata=schema(min=-6))
            high: int = field(default=-2, metadata=schema(min=-6))

        @type_name("ChemPropRegressorWarmup_Epochs_Ratio")
        @dataclass
        class ChemPropParametersWarmup_Epochs_Ratio:
            low: float = field(default=0.1, metadata=schema(min=0.1))
            high: float = field(default=0.1, metadata=schema(min=0.1))
            q: float = field(default=0.1, metadata=schema(min=0.1))

        ensemble_size: Annotated[
            int,
            schema(
                min=1,
                max=5,
                title="Ensemble size",
                description="Number of ensembles with different weight initialisation (provides uncertainty)",
            ),
        ] = field(default=1)

        epochs: Annotated[
            int,
            schema(
                min=4,
                max=400,
                title="Epochs",
                description="Number of epochs to run (increasing this will increase run time)",
            ),
        ] = field(default=30)

        activation: Annotated[
            List[ChemPropActivation],
            schema(
                title="activation",
                description="Activation function applied to the "
                "output of the weighted sum of inputs",
            ),
        ] = field(default_factory=lambda: [ChemPropActivation.RELU])

        aggregation: Annotated[
            List[ChemPropAggregation],
            schema(
                title="aggregation",
                description="Aggregation scheme for atomic vectors into molecular vectors.",
            ),
        ] = field(default_factory=lambda: [ChemPropAggregation.MEAN])

        aggregation_norm: Annotated[
            ChemPropParametersAggregation_Norm,
            schema(
                title="aggregation_norm",
                description="For norm aggregation, number by which to divide summed up atomic features.",
            ),
        ] = ChemPropParametersAggregation_Norm()

        batch_size: Annotated[
            ChemPropParametersBatch_Size,
            schema(
                title="batch_size",
                description="How many samples per batch to load.",
            ),
        ] = ChemPropParametersBatch_Size()

        depth: Annotated[
            ChemPropParametersDepth,
            schema(
                title="depth",
                description="Number of message passing steps"
                "(distance of neighboring atoms visible when modelling).",
            ),
        ] = ChemPropParametersDepth()

        dropout: Annotated[
            ChemPropParametersDropout,
            schema(
                title="dropout",
                description="Dropout probability. During training, randomly zeroes"
                " some of the elements of the input tensor with probability `p`"
                " using samples from a Bernoulli distribution. Each channel will"
                " be zeroed out independently on every forward call.  This has"
                " proven to be an effective technique for regularization and "
                "preventing the co-adaptation of neurons",
            ),
        ] = ChemPropParametersDropout()

        features_generator: Annotated[
            List[ChemPropFeatures_Generator],
            schema(
                title="features_generator",
                description="Method of generating additional features.",
            ),
        ] = field(default_factory=lambda: [ChemPropFeatures_Generator.NONE])

        ffn_hidden_size: Annotated[
            ChemPropParametersFFN_Hidden_Size,
            schema(
                title="ffn_hidden_size",
                description="Dimensionality of hidden layers in the FFN.",
            ),
        ] = ChemPropParametersFFN_Hidden_Size()

        ffn_num_layers: Annotated[
            ChemPropParametersFFN_Num_Layers,
            schema(
                title="ffn_num_layers",
                description="Number of layers in the FFN after D-MPNN encoding.",
            ),
        ] = ChemPropParametersFFN_Num_Layers()

        final_lr_ratio_exp: Annotated[
            ChemPropParametersFinal_Lr_Ratio_Exp,
            schema(
                title="final_lr_ratio_exp",
                description="The exponential for the final learning rate.",
            ),
        ] = ChemPropParametersFinal_Lr_Ratio_Exp()

        hidden_size: Annotated[
            ChemPropParametersHidden_Size,
            schema(
                title="hidden_size",
                description="Size of the hidden bond message vectors in the D-MPNN",
            ),
        ] = ChemPropParametersHidden_Size()

        init_lr_ratio_exp: Annotated[
            ChemPropParametersInit_Lr_Ratio_Exp,
            schema(
                title="init_lr_ratio_exp",
                description="The exponential for the learning rate ratio.",
            ),
        ] = ChemPropParametersInit_Lr_Ratio_Exp()

        max_lr_exp: Annotated[
            ChemPropParametersMax_Lr_Exp,
            schema(
                title="max_lr_exp",
                description="The exponential for the maximum learning rate.",
            ),
        ] = ChemPropParametersMax_Lr_Exp()

        warmup_epochs_ratio: Annotated[
            ChemPropParametersWarmup_Epochs_Ratio,
            schema(
                title="warmup_epochs_ratio",
                description="Ratio for the number of epochs during which "
                "learning rate increases linearly from init_lr to max_lr."
                " Afterwards, learning rate decreases exponentially "
                "from max_lr to final_lr.",
            ),
        ] = ChemPropParametersWarmup_Epochs_Ratio()

    name: Literal["ChemPropRegressor"]
    parameters: Parameters


class ChemPropFrzn(str, Enum):
    """
    `QSARtuna` implements a hyperparameter search space level for ChemProp in order to define Hyperopt search space
     to optimise. Increasing levels correspond to increasing the search space.
    """

    NONE = "none"
    """No weights are frozen"""
    MPNN = "mpnn"
    """Freeze the weights in only the MPNN during transfer learning"""
    MPNN_FIRST_FFN = "mpnn_first_ffn"
    """Freeze the MPNN and first layer of the FFN during transfer learning"""
    MPNN_LAST_FFN = "mpnn_last_ffn"
    """Freeze the MPNN and until the penultimate layer of the FFN during transfer learning"""


@dataclass
class ChemPropRegressorPretrained(Algorithm):
    """Chemprop Regressor from a pretrined model

    Pretraining can be carried out by supplying previously trained QSARtuna ChemProp PKL model.
    """

    @type_name("ChemPropRegressorPretrainedParams")
    @dataclass
    class Parameters:
        @type_name("ChemPropRegressorPretrainedEpochs")
        @dataclass
        class ChemPropParametersEpochs:
            low: int = field(default=4, metadata=schema(min=0, max=300))
            high: int = field(default=30, metadata=schema(min=0, max=300))
            q: int = field(default=1, metadata=schema(min=1))

        epochs: Annotated[
            ChemPropParametersEpochs,
            schema(
                title="epochs",
                description="Number of epochs to fine-tune the pretrained model on new data",
            ),
        ] = ChemPropParametersEpochs()

        frzn: Annotated[
            List[ChemPropFrzn],
            schema(
                title="Frozen layers",
                description="Decide which layers of the MPNN or FFN to freeze during transfer learning.",
            ),
        ] = field(default_factory=lambda: [ChemPropFrzn.NONE])

        pretrained_model: Annotated[
            str,
            schema(
                title="Pretrained Model",
                description="Path to a pretrained QSARtuna pkl model",
            ),
        ] = field(default=None)

    name: Literal["ChemPropRegressorPretrained"]
    parameters: Parameters


class ChemPropSearch_Parameter_Level(str, Enum):
    """
    `QSARtuna` implements a hyperparameter search space level for ChemProp in order to define Hyperopt search space
     to optimise. Increasing levels correspond to increasing the search space.
    """

    AUTO = "auto"
    """Alter the space depending on training data, i.e training set size, no. of hyperparameter trial configurations 
    (`num_iters`) & no. epochs. This ensures search spaces are not too large for limited data/epochs, 
    and `vice-versa`, an extensive search space is trailed when applicable."""
    L1 = "1"
    """Search only the `basic` set of hyperparameters: `depth`, `ffn_num_layers`, `dropout`, and
     `linked_hidden_size`."""
    L2 = "2"
    """Search `basic` and `linked_hidden_size` (search `hidden_size` and `ffn_hidden_size` constrained to have the
     same value."""
    L3 = "3"
    """Search for `basic`, "hidden_size", "ffn_hidden_size` and `learning_rate` (which uses `max_lr_exp`, `init_lr_exp`,
     `final_lr_exp`, and `warmup_epochs` parameters)."""
    L4 = "4"
    """Search for `basic`, `hidden_size`, `ffn_hidden_size` (hidden sizes now independent) and `learning_rate`."""
    L5 = "5"
    """Search for `basic`, `hidden_size`, `ffn_hidden_size`, `learning_rate` and `activation`."""
    L6 = "6"
    """Search for `basic`, `hidden_size`, `ffn_hidden_size`, `learning_rate`, `activation` and `batch_size`."""
    L7 = "7"
    """Search for `basic`, `hidden_size`, `ffn_hidden_size`, `learning_rate`, `activation`, `batch_size` and
    `aggregation_norm`."""
    L8 = "8"
    """Search all possible `network` hyper-parameters"""


@dataclass
class ChemPropHyperoptClassifier(Algorithm):
    """Chemprop classifier

    Chemprop is an open-source package for training deep learning models for molecular property prediction. ChemProp
    trains two networks; a Directed Message Passing Neural Network (D-MPNN) to encode a graph representation of
    molecules, and a Feed Forward Neural Network (FFNN); a standard multi-layer perceptron trained to predict the
    target property using D-MPNN encoding. It was first presented in the paper "Analyzing Learned Molecular
    Representations for Property Prediction".  This implementation will use Hyperopt to optimse `network` parameters
    within each trial, allowing for Optuna to trial more complex hyperparameters, such as feature generation
    and side information weighting. NB: This implementation can also be used to implement quick/simple ChemProp models
    by using sensible defaults from the authors; to do this run ChemProp with Num_Iters='1'.
    """

    @type_name("ChemPropHyperoptClassifierParams")
    @dataclass
    class Parameters:
        ensemble_size: Annotated[
            int,
            schema(
                min=1,
                max=5,
                title="Ensemble size",
                description="Number of ensembles with different weight initialisation (provides uncertainty)",
            ),
        ] = field(default=1)

        epochs: Annotated[
            int,
            schema(
                min=4,
                max=400,
                title="Epochs",
                description="Number of epochs to run (increasing this will increase run time)",
            ),
        ] = field(default=30)

        num_iters: Annotated[
            int,
            schema(
                min=1,
                max=50,
                title="Number of HyperOpt iterations",
                description="Dictates the number (Hyperopt) trials ChemProp will run",
            ),
        ] = field(default=1)

        features_generator: Annotated[
            List[ChemPropFeatures_Generator],
            schema(
                title="features_generator",
                description="Method of generating additional features.",
            ),
        ] = field(default_factory=lambda: [ChemPropFeatures_Generator.NONE])

        search_parameter_level: Annotated[
            List[ChemPropSearch_Parameter_Level],
            schema(
                title="search_parameter_level",
                description="Defines the complexity of the search space used by Hyperopt (larger=more complex).",
            ),
        ] = field(default_factory=lambda: [ChemPropSearch_Parameter_Level.AUTO])

    name: Literal["ChemPropHyperoptClassifier"]
    parameters: Parameters


@dataclass
class ChemPropHyperoptRegressor(Algorithm):
    """Chemprop regressor

    Chemprop is an open-source package for training deep learning models for molecular property prediction. ChemProp
    trains two networks; a Directed Message Passing Neural Network (D-MPNN) to encode a graph representation of
    molecules, and a Feed Forward Neural Network (FFNN); a standard multi-layer perceptron trained to predict the
    target property using D-MPNN encoding. It was first presented in the paper "Analyzing Learned Molecular
    Representations for Property Prediction".  This implementation will use Hyperopt to optimse `network` parameters
    within each trial, allowing for Optuna to trial more complex hyperparameters, such as feature generation
    and side information weighting. NB: This implementation can also be used to implement quick/simple ChemProp models
    by using sensible defaults from the authors; to do this run ChemProp with Num_Iters='1'.
    """

    @type_name("ChemPropHyperoptRegressorParams")
    @dataclass
    class Parameters:
        ensemble_size: Annotated[
            int,
            schema(
                min=1,
                max=5,
                title="Ensemble size",
                description="Number of ensembles with different weight initialisation (provides uncertainty)",
            ),
        ] = field(default=1)

        epochs: Annotated[
            int,
            schema(
                min=4,
                max=400,
                title="Epochs",
                description="Number of epochs to run (increasing this will increase run time)",
            ),
        ] = field(default=30)

        num_iters: Annotated[
            int,
            schema(
                min=1,
                max=50,
                title="Number of HyperOpt iterations",
                description="Dictates the number (Hyperopt) trials ChemProp will run",
            ),
        ] = field(default=1)

        features_generator: Annotated[
            List[ChemPropFeatures_Generator],
            schema(
                title="features_generator",
                description="Method of generating additional features.",
            ),
        ] = field(default_factory=lambda: [ChemPropFeatures_Generator.NONE])

        search_parameter_level: Annotated[
            List[ChemPropSearch_Parameter_Level],
            schema(
                title="search_parameter_level",
                description="Defines the complexity of the search space used by Hyperopt (larger=more complex).",
            ),
        ] = field(default_factory=lambda: [ChemPropSearch_Parameter_Level.AUTO])

    name: Literal["ChemPropHyperoptRegressor"]
    parameters: Parameters


@dataclass
class CustomClassificationModel(Algorithm):
    """Classifier  from a preexisting pkl model"""

    @type_name("CustomClassificationModelParams")
    @dataclass
    class Parameters:
        preexisting_model: Annotated[
            str,
            schema(
                title="Preexisting Model",
                description="Path to a preexisting pkl model",
            ),
        ] = field(default=None)
        refit_model: Annotated[
            int,
            schema(
                min=0,
                max=1,
                title="Refit Model",
                description="Whether fit should be called during the trial of the custom model",
            ),
        ] = field(default=0)

    name: Literal["CustomClassificationModel"]
    parameters: Parameters


@dataclass
class CustomRegressionModel(Algorithm):
    """Classifier  from a preexisting pkl model"""

    @type_name("CustomRegressionModelParams")
    @dataclass
    class Parameters:
        preexisting_model: Annotated[
            str,
            schema(
                title="Preexisting Model",
                description="Path to a preexisting pkl model",
            ),
        ] = field(default=None)
        refit_model: Annotated[
            int,
            schema(
                min=0,
                max=1,
                title="Refit Model",
                description="Whether fit should be called during the trial of the custom model",
            ),
        ] = field(default=0)

    name: Literal["CustomRegressionModel"]
    parameters: Parameters


AnyRegressionAlgorithm = Union[
    Lasso,
    PLSRegression,
    RandomForestRegressor,
    Ridge,
    KNeighborsRegressor,
    SVR,
    XGBRegressor,
    PRFClassifier,  # PRFClassifier ingests/outputs continuous probabilities so should be evaluated as regressor
    ChemPropRegressor,
    ChemPropRegressorPretrained,
    ChemPropHyperoptRegressor,
    CustomRegressionModel,
]

AnyClassificationAlgorithm = Union[
    AdaBoostClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    ChemPropClassifier,
    ChemPropHyperoptClassifier,
    CustomClassificationModel,
]


class CalibratedClassifierCVEnsemble(str, Enum):
    TRUE = "True"
    FALSE = "False"


class CalibratedClassifierCVMethod(str, Enum):
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"
    VENNABERS = "vennabers"


@dataclass
class CalibratedClassifierCVWithVA(Algorithm):
    """Calibrated Classifier.

    Probability calibration with isotonic regression, logistic regression, or VennABERS.

    This class uses cross-validation (cv) to both estimate the parameters of a classifier and subsequently calibrate
    a classifier. With default ensemble=True, for each cv split it fits a copy of the base estimator to the training
    subset, and calibrates it using the testing subset. For prediction, predicted probabilities are averaged across
    these individual calibrated classifiers. When ensemble=False, cv is used to obtain unbiased predictions which are
    then used for calibration. For prediction, the base estimator, trained using all the data, is used. VennABERS
    offers uncertainty prediction based on p0 vs. p1 discordance.
    """

    @type_name("CalibratedClassifierCVWithVAParams")
    @dataclass
    class Parameters:
        estimator: Annotated[
            AnyClassificationAlgorithm,
            schema(
                title="Estimator",
                description="Base estimator to use for calibration",
            ),
        ] = AnyClassificationAlgorithm

        ensemble: Annotated[
            Union[CalibratedClassifierCVEnsemble, str],
            schema(
                title="ensemble",
                description="Whether each cv it fits a copy of the base estimator, vs. cv used to obtain unbiased "
                "predictions used for calibration",
            ),
        ] = field(default=CalibratedClassifierCVEnsemble.TRUE)

        method: Annotated[
            Union[CalibratedClassifierCVMethod, str],
            schema(
                title="method",
                description="Calibration method used to obtained calibrated predictions",
            ),
        ] = field(default=CalibratedClassifierCVMethod.ISOTONIC)
        n_folds: Annotated[
            int,
            schema(
                min=2,
                max=5,
                title="Number of Cross validation folds (splits)",
                description="Number of cv folds to obtain calibration data",
            ),
        ] = field(default=2)

    name: Literal["CalibratedClassifierCVWithVA"]
    parameters: Parameters


MapieCompatibleAlgorithm = Union[
    Lasso,
    PLSRegression,
    RandomForestRegressor,
    KNeighborsRegressor,
    Ridge,
    SVR,
    XGBRegressor,
    PRFClassifier,
]


@dataclass
class Mapie(Algorithm):
    """Mapie

    MAPIE - Model Agnostic Prediction Interval Estimator

    MAPIE allows you to estimate prediction intervals for regression models. Prediction intervals output by MAPIE
    encompass both aleatoric and epistemic uncertainties and are backed by strong theoretical guarantees thanks to
    conformal prediction methods.
    """

    @type_name("MapieParams")
    @dataclass
    class Parameters:
        estimator: Annotated[
            MapieCompatibleAlgorithm,
            schema(
                title="Estimator",
                description="Base estimator to use",
            ),
        ] = MapieCompatibleAlgorithm
        mapie_alpha: Annotated[
            float,
            schema(
                min=0.01,
                max=0.99,
                title="Uncertainty alpha",
                description="Alpha used to generate uncertainty estimates",
            ),
        ] = field(default=0.05)

    name: Literal["Mapie"]
    parameters: Parameters


AnyAlgorithm = Union[
    AnyRegressionAlgorithm,
    AnyClassificationAlgorithm,
    CalibratedClassifierCVWithVA,
    Mapie,
]

AnyChemPropAlgorithm = [
    ChemPropClassifier,
    ChemPropHyperoptClassifier,
    ChemPropRegressor,
    ChemPropRegressorPretrained,
    ChemPropHyperoptRegressor,
]


def isanyof(obj: Any, classes: Iterable[Type]) -> bool:
    return any(isinstance(obj, cls) for cls in classes)


def detect_mode_from_algs(algs: List[AnyAlgorithm]) -> ModelMode:
    # Getting algs could be replaced by
    # TO DO: typing.get_args(AnyRegressionAlgorithm) from Python 3.8.
    regression_algs = [
        Lasso,
        PLSRegression,
        RandomForestRegressor,
        Ridge,
        SVR,
        XGBRegressor,
        ChemPropRegressor,
        ChemPropRegressorPretrained,
        ChemPropHyperoptRegressor,
        PRFClassifier,
        Mapie,
        CustomRegressionModel,
        KNeighborsRegressor
    ]
    classification_algs = [
        AdaBoostClassifier,
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        ChemPropClassifier,
        ChemPropHyperoptClassifier,
        CalibratedClassifierCVWithVA,
        CustomClassificationModel,
        KNeighborsClassifier
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
    descriptors: List[MolDescriptor], dataset: Dataset, cache: Optional[Memory]
) -> None:
    def recursive_copy_path_for_scaled_descriptor(d_: MolDescriptor) -> None:
        if isinstance(d_, ScaledDescriptor):
            scaler = d_.parameters.scaler
            if (
                isinstance(scaler, UnfittedSklearnScaler)
                and scaler.mol_data.file_path is None
            ):
                d_.set_unfitted_scaler_data(
                    dataset.training_dataset_file, dataset.input_column, cache=cache
                )
            else:
                d_._ensure_scaler_is_fitted(cache=cache)
        elif isinstance(d_, CompositeDescriptor):
            for dd in d_.parameters.descriptors:
                try:
                    recursive_copy_path_for_scaled_descriptor(dd)
                except ScalingFittingError:
                    logger.info(
                        f"{dd} failed scaling, will not be used in optimisation"
                    )

    for d in descriptors:
        try:
            recursive_copy_path_for_scaled_descriptor(d)
        except ScalingFittingError:
            logger.info(f"{d} failed scaling, will not be used in optimisation")


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
            default=None,
            metadata=schema(title="Classification or regression.") | none_as_undefined,
        )

        cross_validation: int = field(
            default=5,
            metadata=schema(
                min=1,
                max=10,
                title="Hyperparameter splits",
                description="Number of cross-validation splits Optuna will use during hyperparameter optimization"
                " Use '1' to disable cross-validation,"
                " but note that KFold requires at least two splits/folds.",
            ),
        )

        cv_split_strategy: AnyCvSplitter = field(
            default_factory=Random,
            metadata=schema(
                title="Hyperparameter split strategy",
                description="Splitting method used by Optuna during hyperparameter optimization",
            ),
        )

        shuffle: bool = field(
            default=False,
            metadata=schema(
                title="Hyperparameter data shuffle",
                description="Whether or not to shuffle the data for Optuna cross-validation",
            ),
        )

        direction: Optional[OptimizationDirection] = field(
            default=OptimizationDirection.MAXIMIZATION,  # Defaults: Sklearn - maximize, Optuna - minimize.
            metadata=schema(title="Maximization or minimization") | none_as_undefined,
        )

        scoring: Union[RegressionScore, ClassificationScore, str] = field(
            default=None,
            metadata=schema(
                title="Scoring metric",
                description="The scoring metric that Optuna will use to optimise",
            )
            | none_as_undefined,
        )

        minimise_std_dev: bool = field(
            default=False,
            metadata=schema(
                title="Minimise cross-fold deviation",
                description="Whether or not to require Optuna to also optimise for low cross-fold standard deviation "
                "of the primary metric",
            ),
        )

        use_cache: bool = field(
            default=True,
            metadata=schema(
                title="Cache descriptor calculations",
                description="Whether or not to allow Optuna to cache descriptor generation for latency improvements",
            ),
        )

        n_trials: Optional[int] = field(
            default=300,
            metadata=schema(
                min=0,  # Zero for no optimization (but run preprocessing).
                title="Hyperparameter trials",
                description="Number of Optuna trials for hyperparameter optimization",
            )
            | none_as_undefined,
        )

        n_jobs: Optional[int] = field(
            default=-1,
            metadata=schema(
                title="Number of parallel jobs, set to '-1' to use as many jobs as CPU cores available"
            )
            | none_as_undefined,
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
                title="Seed for reproducibility",
                description="This seed is used for a random number generator."
                " Set to an integer value to get reproducible results."
                " Set to None/null/empty to initialize at random.",
            )
            | none_as_undefined,
        )

        optuna_storage: Optional[str] = field(
            default=None,
            metadata=schema(
                title="Database URL for Optuna Storage",
                description="Database URL for Optuna Storage."
                " Set to None/null to use default in-memory storage."
                " Example: 'sqlite:///optuna_storage.db' for file-based SQLite3 storage.",
            )
            | none_as_undefined,
        )

        track_to_mlflow: Optional[bool] = field(
            default=False,
            metadata=schema(
                title="Track to MLFlow",
                description="Set to True to use MLFlow tracking UI,"
                " set to False to disable MLFlow.",
            ),
        )

        tracking_rest_endpoint: Optional[str] = field(
            default=None,
            metadata=schema(title="URL to track Optuna progress using internal format"),
        )

        split_chemprop: Optional[bool] = field(
            default=True,
            metadata=schema(
                title="Whether or not to split ChemProp into separate Optuna runs"
            ),
        )

        n_chemprop_trials: Optional[int] = field(
            default=1,
            metadata=schema(
                title="Number of ChemProp Optuna runs",
                description="Dictates the number of optimization runs to perform using Optuna. Will only be used if "
                "ChemProp is supplied as an algorithm ",
            ),
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
    data: Dataset = field(
        default=None,
        metadata=schema(
            title="Dataset", description="Input data and preprocessing steps."
        )
        | none_as_undefined
        | required,
    )
    mode: Optional[ModelMode] = field(
        default=None,
        metadata=schema(title="Classification or regression") | none_as_undefined,
    )  # For GUI compatibility.
    algorithms: List[AnyAlgorithm] = field(
        default=None,
        metadata=schema(
            title="Algorithms",
            description="Algorithms to trial during hyperparameter optimization.",
            min_items=1,
        )
        | none_as_undefined
        | required,
    )
    descriptors: List[MolDescriptor] = field(
        default=None,
        metadata=schema(
            title="Descriptors",
            description="Molecular descriptors to trial during hyperparameter optimizations.",
            min_items=1,
        )
        | none_as_undefined
        | required,
    )
    settings: Settings = field(
        default=None,
        metadata=schema(
            title="Settings", description="Detailed settings for the optimization job."
        ),
    )
    visualization: Optional[Visualization] = field(default=None)
    task: Literal["optimization"] = "optimization"

    def set_cache(self):
        """Set the cache for descriptor generation when the number of cores supports this"""
        if hasattr(self, "_cache") and self._cache is not None:
            logger.warning(f"cache already set.")
        else:
            cachedir = TemporaryDirectory()
            memory = Memory(cachedir.name, verbose=0)
            memory.n_cores = self.settings.n_jobs
            self._cache = memory
            self._cache_dir = cachedir

    def set_algo_hashes(self):
        """Set hashes for the algorithms

        This facilitates tracking duplicate algorithm types with distinct param setups
        """
        for algorithm in self.algorithms:
            algorithm.hash = md5_hash(serialize(algorithm))
            if hasattr(algorithm.parameters, "estimator"):
                algorithm.parameters.estimator.hash = md5_hash(serialize(algorithm))

    def __post_init__(self):
        # Sync 'mode' in "root" (for GUI) and in settings.mode (original).
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

        # Set the cache at post init if True in settings
        if self.settings.use_cache:
            self.set_cache()
        else:
            # set cache attributes to None, until an (optional) future call to self.set_cache()
            self._cache = None
            self._cache_dir = None

        # Hash the algorithm options (to allow for trialing duplicate algorithms within one config)
        self.set_algo_hashes()

        # Tell scaled descriptor to use the main dataset by default.
        copy_path_for_scaled_descriptor(self.descriptors, self.data, self._cache)

        # Set default scoring.
        if self.settings.scoring is None:
            if self.settings.mode == ModelMode.REGRESSION:
                self.settings.scoring = RegressionScore.NEG_MEAN_SQUARED_ERROR
            elif self.settings.mode == ModelMode.CLASSIFICATION:
                self.settings.scoring = ClassificationScore.ROC_AUC
        elif isinstance(self.settings.scoring, Enum):
            self.settings.scoring = self.settings.scoring.value

        # Set default response type.
        if self.mode == ModelMode.REGRESSION:
            self.data.response_type = "regression"
        elif self.mode == ModelMode.CLASSIFICATION:
            self.data.response_type = "classification"
