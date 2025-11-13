import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional, Union

from apischema import schema, serialize, type_name
from apischema.metadata import none_as_undefined, required
from joblib import Memory

from optunaz.config import Algorithm as GenericAlg
from optunaz.config import ModelMode, OptimizationDirection, Visualization
from optunaz.datareader import Dataset
from optunaz.descriptors import MolDescriptor
from optunaz.utils import md5_hash
from optunaz.utils.files_paths import copy_path_for_scaled_descriptor
from optunaz.utils.preprocessing.splitter import AnyCvSplitter, Stratified

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
            low: float = field(default=0.001, metadata=schema(title="low", min=0.0001))
            high: float = field(default=2.0, metadata=schema(title="high", min=0.001))

        @dataclass
        class AdaBoostClassifierParametersMaxDepth:
            low: int = field(default=1, metadata=schema(title="low", min=1))
            high: int = field(default=3, metadata=schema(title="high", min=1))

        n_estimators: AdaBoostClassifierParametersNEstimators = field(
            default_factory=AdaBoostClassifierParametersNEstimators,
            metadata=schema(
                title="n_estimators",
                description="The maximum number of estimators"
                " at which boosting is terminated."
                " In case of perfect fit, the learning procedure is stopped early.",
            ),
        )
        learning_rate: AdaBoostClassifierParametersLearningRate = field(
            default_factory=AdaBoostClassifierParametersLearningRate,
            metadata=schema(
                title="learning_rate",
                description="Weight applied to each classifier"
                "at each boosting iteration. A higher learning rate"
                "increases the contribution of each classifier. "
                "There is a trade-off between the learning_rate"
                "and n_estimators parameters.",
            ),
        )
        max_depth: AdaBoostClassifierParametersMaxDepth = field(
            default_factory=AdaBoostClassifierParametersMaxDepth,
            metadata=schema(
                title="max_depth",
                description="Maximum depth of the individual decision tree estimators."
                " The maximum depth limits the number of nodes in the tree,"
                " which may reduce overfitting. Larger values may yield more predictive models.",
            ),
        )

    name: Literal["AdaBoostClassifier"] = "AdaBoostClassifier"
    parameters: Parameters = field(default_factory=Parameters)


@dataclass
class CatBoostClassifier(Algorithm):
    """CatBoostClassifier Classifier.

    A CatBoostClassifier classifier is a gradient boosting algorithm
    that uses decision trees as base learners. The algorithm is designed to handle categorical features
    and is known for its robustness against overfitting and ability to handle large datasets efficiently.
    This implementation supports GPU acceleration and is particularly efficient for large datasets.

    CatBoost incorporates several algorithmic techniques (like ordered boosting and minimal variance sampling)
    aimed at reducing overfitting, especially on small datasets. Both frameworks include built-in
    cross-validation and regularization knobs. CatBoost’s architecture can often give superior performance
    when overfitting is a concern.
    """

    @type_name("CatboostClassifierParams")
    @dataclass
    class Parameters:
        @dataclass
        class CatboostClassifierParametersNEstimators:
            low: int = field(default=500, metadata=schema(title="low", min=1))
            high: int = field(default=500, metadata=schema(title="high", min=1))

        @dataclass
        class CatboostClassifierParametersLearningRate:
            low: float = field(default=0.1, metadata=schema(title="low", min=0.0001))
            high: float = field(default=0.3, metadata=schema(title="high", min=0.001))

        @dataclass
        class CatboostClassifierParametersDepth:
            low: int = field(default=3, metadata=schema(title="low", min=1))
            high: int = field(default=8, metadata=schema(title="high", min=1))

        @dataclass
        class CatboostClassifierParametersL2LeafReg:
            low: float = field(default=1e-2, metadata=schema(title="low", min=0.0001))
            high: float = field(default=10.0, metadata=schema(title="high", min=0.0001))

        @dataclass
        class CatboostClassifierParametersRandomStrength:
            low: float = field(default=1e-2, metadata=schema(title="low", min=0.0001))
            high: float = field(default=10.0, metadata=schema(title="high", min=0.0001))

        n_estimators: CatboostClassifierParametersNEstimators = field(
            default_factory=CatboostClassifierParametersNEstimators,
            metadata=schema(
                title="Number Estimators",
                description="The maximum number of estimators"
                " at which boosting is terminated."
                " In case of perfect fit, the learning procedure is stopped early.",
            ),
        )
        learning_rate: CatboostClassifierParametersLearningRate = field(
            default_factory=CatboostClassifierParametersLearningRate,
            metadata=schema(
                title="Learning Rate",
                description="Weight applied to each classifier"
                "at each boosting iteration. A higher learning rate"
                "increases the contribution of each classifier. "
                "There is a trade-off between the learning_rate"
                "and n_estimators parameters.",
            ),
        )
        depth: CatboostClassifierParametersDepth = field(
            default_factory=CatboostClassifierParametersDepth,
            metadata=schema(
                title="Depth", description="The depth of each boosting iteration"
            ),
        )
        l2_leaf_reg: CatboostClassifierParametersL2LeafReg = field(
            default_factory=CatboostClassifierParametersL2LeafReg,
            metadata=schema(
                title="L2 Leaf", description="The L2 Leaf for each boosting iteration"
            ),
        )
        random_strength: CatboostClassifierParametersRandomStrength = field(
            default_factory=CatboostClassifierParametersRandomStrength,
            metadata=schema(
                title="Random Strength",
                description="The random strength for each boosting iteration."
                " This controls the amount of randomness that is introduced"
                " when finding the best splits during tree construction. It"
                " influences both regularization and performance",
            ),
        )

    name: Literal["CatBoostClassifier"] = "CatBoostClassifier"
    parameters: Parameters = field(default_factory=Parameters)


@dataclass
class CatBoostRegressor(Algorithm):
    """CatBoostRegressor Regressor.

    A CatBoostRegressor regressor is a gradient boosting algorithm
    that uses decision trees as base learners. The algorithm is designed to handle categorical features
    and is known for its robustness against overfitting and ability to handle large datasets efficiently.
    This implementation supports GPU acceleration and is particularly efficient for large datasets.

    CatBoost incorporates several algorithmic techniques (like ordered boosting and minimal variance sampling)
    aimed at reducing overfitting, especially on small datasets. Both frameworks include built-in
    cross-validation and regularization knobs. CatBoost’s architecture can often give superior performance
    when overfitting is a concern.
    """

    @type_name("CatboostRegressorParams")
    @dataclass
    class Parameters:
        @dataclass
        class CatboostRegressorParametersNEstimators:
            low: int = field(default=500, metadata=schema(title="low", min=1))
            high: int = field(default=500, metadata=schema(title="high", min=1))

        @dataclass
        class CatboostRegressorParametersLearningRate:
            low: float = field(default=0.1, metadata=schema(title="low", min=0.0001))
            high: float = field(default=0.3, metadata=schema(title="high", min=0.001))

        @dataclass
        class CatboostRegressorParametersDepth:
            low: int = field(default=3, metadata=schema(title="low", min=1))
            high: int = field(default=8, metadata=schema(title="high", min=1))

        @dataclass
        class CatboostRegressorParametersL2LeafReg:
            low: float = field(default=1e-2, metadata=schema(title="low", min=0.0001))
            high: float = field(default=10.0, metadata=schema(title="high", min=0.0001))

        @dataclass
        class CatboostRegressorParametersRandomStrength:
            low: float = field(default=1e-2, metadata=schema(title="low", min=0.0001))
            high: float = field(default=10.0, metadata=schema(title="high", min=0.0001))

        n_estimators: CatboostRegressorParametersNEstimators = field(
            default_factory=CatboostRegressorParametersNEstimators,
            metadata=schema(
                title="Number Estimators",
                description="The maximum number of estimators"
                " at which boosting is terminated."
                " In case of perfect fit, the learning procedure is stopped early.",
            ),
        )
        learning_rate: CatboostRegressorParametersLearningRate = field(
            default_factory=CatboostRegressorParametersLearningRate,
            metadata=schema(
                title="Learning Rate",
                description="Weight applied to each regressor"
                "at each boosting iteration. A higher learning rate"
                "increases the contribution of each regressor. "
                "There is a trade-off between the learning_rate"
                "and n_estimators parameters.",
            ),
        )
        depth: CatboostRegressorParametersDepth = field(
            default_factory=CatboostRegressorParametersDepth,
            metadata=schema(
                title="Depth", description="The depth of each boosting iteration"
            ),
        )
        l2_leaf_reg: CatboostRegressorParametersL2LeafReg = field(
            default_factory=CatboostRegressorParametersL2LeafReg,
            metadata=schema(
                title="L2 Leaf", description="The L2 Leaf for each boosting iteration"
            ),
        )
        random_strength: CatboostRegressorParametersRandomStrength = field(
            default_factory=CatboostRegressorParametersRandomStrength,
            metadata=schema(
                title="Random Strength",
                description="The random strength for each boosting iteration."
                " This controls the amount of randomness that is introduced"
                " when finding the best splits during tree construction. It"
                " influences both regularization and performance",
            ),
        )

    name: Literal["CatBoostRegressor"] = "CatBoostRegressor"
    parameters: Parameters = field(default_factory=Parameters)


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
            low: float = field(default=1e-5, metadata=schema(title="low", min=1e-5))
            high: float = field(default=2.0, metadata=schema(title="high", min=1e-5))

        @dataclass
        class LassoParametersMaxIter:
            low: int = field(default=1000, metadata=schema(title="low", min=1000))
            high: int = field(default=10000, metadata=schema(title="high", min=1000))
            step: int = field(default=1000, metadata=schema(min=100))

        @dataclass
        class LassoParametersTol:
            low: float = field(default=1e-6, metadata=schema(title="low", min=1e-6))
            high: float = field(default=1e-2, metadata=schema(title="high", min=1e-2))

        alpha: LassoParametersAlpha = field(
            default_factory=LassoParametersAlpha,
            metadata=schema(
                title="Alpha",
                description="Constant that multiplies the L1 term,"
                " controlling regularization strength.",
            ),
        )
        max_iter: LassoParametersMaxIter = field(
            default_factory=LassoParametersMaxIter,
            metadata=schema(
                title="Max Iterations",
                description="The maximum number of iterations to run the optimization algorithm."
                " The optimization is stopped when either the maximum number of iterations"
                " is reached or the optimization converges.",
            ),
        )
        tol: LassoParametersTol = field(
            default_factory=LassoParametersTol,
            metadata=schema(
                title="Tol",
                description="Precision of the optimization algorithm."
                " The optimization is stopped when the change in the loss function"
                " is less than this value.",
            ),
        )

    name: Literal["Lasso"] = "Lasso"
    parameters: Parameters = field(default_factory=Parameters)


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

        n_neighbors: KNeighborsClassifierParametersN_Neighbors = field(
            default_factory=KNeighborsClassifierParametersN_Neighbors,
            metadata=schema(
                title="N Neighbors",
                description="Number of neighbors to use by default for kneighbors queries.",
            ),
        )
        weights: List[KNeighborsWeights] = field(
            default_factory=lambda: [KNeighborsWeights.UNIFORM],
            metadata=schema(
                title="Weights",
                description="Weight function used in prediction",
            ),
        )
        metric: List[KNeighborsMetric] = field(
            default_factory=lambda: [KNeighborsMetric.MINKOWSKI],
            metadata=schema(
                title="Metric",
                description="Metric to use for distance computation."
                "The default of “minkowski” results in the standard Euclidean distance",
            ),
        )

    name: Literal["KNeighborsClassifier"] = "KNeighborsClassifier"
    parameters: Parameters = field(default_factory=Parameters)


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

        n_neighbors: KNeighborsRegressorParametersN_Neighbors = field(
            default_factory=KNeighborsRegressorParametersN_Neighbors,
            metadata=schema(
                title="N Neighbors",
                description="Number of neighbors to use by default for kneighbors queries.",
            ),
        )
        weights: List[KNeighborsWeights] = field(
            default_factory=lambda: [KNeighborsWeights.UNIFORM],
            metadata=schema(
                title="Weights",
                description="Weight function used in prediction",
            ),
        )
        metric: List[KNeighborsMetric] = field(
            default_factory=lambda: [KNeighborsMetric.MINKOWSKI],
            metadata=schema(
                title="Metric",
                description="Metric to use for distance computation."
                "The default of “minkowski” results in the standard Euclidean distance",
            ),
        )

    name: Literal["KNeighborsRegressor"] = "KNeighborsRegressor"
    parameters: Parameters = field(default_factory=Parameters)


class LogisticRegressionPenalty(str, Enum):
    """Penalty used for the LogisticRegression Classifier"""

    L1 = "l1"
    L2 = "l2"


class LogisticRegressionSolver(str, Enum):
    """Penalty used for the LogisticRegression Classifier"""

    NEWTON_CG = "newton-cg"
    LBFGS = "lbfgs"
    SAG = "sag"
    SAGA = "saga"


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
            high: float = field(default=1000, metadata=schema(title="high", max=1000))

        solver: List[LogisticRegressionSolver] = field(
            default_factory=lambda: [LogisticRegressionSolver.LBFGS],
            metadata=schema(
                title="Solver",
                description="Detemines the specific optimization algorithm used to"
                " find the best-fitting parameters for the model; some solvers"
                " (like ‘liblinear’) work well for smaller datasets or support"
                " only certain types of penalties, while others (such as ‘lbfgs’ or ‘saga’)"
                " are better suited for larger datasets and can handle more complex"
                " regularization schemes. Choosing the right solver ensures training is"
                " both efficient and appropriate for your data and chosen penalty.",
            ),
        )
        C: LogisticRegressionParametersParameterC = field(
            default_factory=LogisticRegressionParametersParameterC,
            metadata=schema(
                title="C",
                description="Inverse of regularization strength;"
                " must be a positive float."
                " Like in support vector machines,"
                " smaller values specify stronger regularization.",
            ),
        )
        penalty: List[LogisticRegressionPenalty] = field(
            default_factory=lambda: [LogisticRegressionPenalty.L2],
            metadata=schema(
                title="Penalty",
                description="The norm used in the penalization."
                " This adds a rule that discourages a models reliance "
                " on any single feature, helping to prevent overfitting and making "
                "predictions more reliable on new data.",
            ),
        )

    name: Literal["LogisticRegression"] = "LogisticRegression"
    parameters: Parameters = field(default_factory=Parameters)


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

        n_components: NComponents = field(
            default_factory=NComponents,
            metadata=schema(
                title="n_components",
                description="Number of components to keep."
                " Should be in [1, min(n_samples, n_features, n_targets)].",
            ),
        )

    name: Literal["PLSRegression"] = "PLSRegression"
    parameters: Parameters = field(default_factory=Parameters)


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
            low: int = field(default=100, metadata=schema(title="low", min=1))
            high: int = field(default=250, metadata=schema(title="high", min=1))

        max_depth: RandomForestClassifierParametersMaxDepth = field(
            default_factory=RandomForestClassifierParametersMaxDepth,
            metadata=schema(
                title="max_depth",
                description="The maximum depth of the tree.",
            ),
        )

        n_estimators: RandomForestClassifierParametersNEstimators = field(
            default_factory=RandomForestClassifierParametersNEstimators,
            metadata=schema(
                title="n_estimators",
                description="The number of trees in the forest.",
            ),
        )

        max_features: List[RandomForestMaxFeatures] = field(
            default_factory=lambda: [RandomForestMaxFeatures.AUTO],
            metadata=schema(
                title="max_features",
                description="The number of features to consider"
                "when looking for the best split:  If auto, then"
                "consider max_features features at each split."
                " - If “auto”, then `max_features=n_features`."
                " - If “sqrt”, then `max_features=sqrt(n_features)`."
                " - If “log2”, then `max_features=log2(n_features)`.",
            ),
        )

    name: Literal["RandomForestClassifier"] = "RandomForestClassifier"
    parameters: Parameters = field(default_factory=Parameters)


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
            low: int = field(default=100, metadata=schema(title="low", min=1))
            high: int = field(default=250, metadata=schema(title="high", min=1))

        max_depth: RandomForestRegressorParametersMaxDepth = field(
            default_factory=RandomForestRegressorParametersMaxDepth,
            metadata=schema(
                title="max_depth",
                description="The maximum depth of the tree.",
            ),
        )

        n_estimators: RandomForestRegressorParametersNEstimators = field(
            default_factory=RandomForestRegressorParametersNEstimators,
            metadata=schema(
                title="n_estimators",
                description="The number of trees in the forest.",
            ),
        )

        max_features: List[RandomForestMaxFeatures] = field(
            default_factory=lambda: [RandomForestMaxFeatures.AUTO],
            metadata=schema(
                title="max_features",
                description="The number of features to consider"
                "when looking for the best split:  If auto, then"
                "consider max_features features at each split."
                " - If “auto”, then `max_features=n_features`."
                " - If “sqrt”, then `max_features=sqrt(n_features)`."
                " - If “log2”, then `max_features=log2(n_features)`.",
            ),
        )

    name: Literal["RandomForestRegressor"] = "RandomForestRegressor"
    parameters: Parameters = field(default_factory=Parameters)


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
            low: float = field(default=0.01, metadata=schema(title="low", min=0.01))
            high: float = field(default=100, metadata=schema(title="high", min=0.01))

        alpha: Alpha = field(
            default_factory=Alpha,
            metadata=schema(
                title="alpha",
                description="Constant that multiplies the L2 term, controlling regularization strength",
            ),
        )

    name: Literal["Ridge"] = "Ridge"
    parameters: Parameters = field(default_factory=Parameters)


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
            low: float = field(default=1e-6, metadata=schema(min=1e-30))
            high: float = field(default=1e06, metadata=schema(max=1e10))

        @dataclass
        class Gamma:
            low: float = field(default=1e-4, metadata=schema(min=1e-9))
            high: float = field(default=1e2, metadata=schema(max=1e3))

        C: SVCParametersParameterC = field(
            default_factory=SVCParametersParameterC,
            metadata=schema(
                title="C",
                description="Regularization parameter."
                " The strength of the regularization is inversely proportional to C."
                " Must be strictly positive."
                " The penalty is a squared l2 penalty.",
            ),
        )

        gamma: Gamma = field(
            default_factory=Gamma,
            metadata=schema(title="gamma", description="Kernel coefficient"),
        )

    name: Literal["SVC"] = "SVC"
    parameters: Parameters = field(default_factory=Parameters)


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
            low: float = field(default=1e-6, metadata=schema(min=1e-30))
            high: float = field(default=1e06, metadata=schema(max=1e10))

        @dataclass
        class SVRParametersGamma:
            low: float = field(default=1e-4, metadata=schema(min=1e-9))
            high: float = field(default=1e2, metadata=schema(max=1e3))

        C: SVRParametersParameterC = field(
            default_factory=SVRParametersParameterC,
            metadata=schema(
                title="C",
                description="Regularization parameter."
                " The strength of the regularization is inversely proportional to C."
                " Must be strictly positive."
                " The penalty is a squared l2 penalty.",
            ),
        )

        gamma: SVRParametersGamma = field(
            default_factory=SVRParametersGamma,
            metadata=schema(title="gamma", description="Kernel coefficient"),
        )

    name: Literal["SVR"] = "SVR"
    parameters: Parameters = field(default_factory=Parameters)


@dataclass
class XGBClassifier(Algorithm):
    """XGBoost classification (gradient boosting trees algorithm).

    XGBoost stands for “Extreme Gradient Boosting”,
    where the term “Gradient Boosting” originates from the paper
    Greedy Function Approximation: A Gradient Boosting Machine, by Friedman.
    """

    @type_name("XGBclassifierParams")
    @dataclass
    class Parameters:
        @dataclass
        class XGBClassifierMaxDepth:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=12, metadata=schema(min=1))

        @dataclass
        class XGBClassifierNEstimators:
            low: int = field(default=100, metadata=schema(min=1))
            high: int = field(default=250, metadata=schema(min=1))

        @dataclass
        class XGBClassifierLearningRate:
            low: float = field(default=0.01, metadata=schema(min=0.0001))
            high: float = field(default=3, metadata=schema(min=0.001))

        @dataclass
        class XGBClassifierSubsample:
            low: float = field(default=0.5, metadata=schema(min=0.01))
            high: float = field(default=1.0, metadata=schema(min=0.01))

        @dataclass
        class XGBClassifierGamma:
            low: float = field(default=1e-3, metadata=schema(min=1e-4))
            high: float = field(default=10, metadata=schema(min=1e-4))

        @dataclass
        class XGBClassifierColSampleByTree:
            low: float = field(default=0.6, metadata=schema(min=0.0))
            high: float = field(default=1.0, metadata=schema(min=0.0))

        max_depth: XGBClassifierMaxDepth = field(
            default_factory=XGBClassifierMaxDepth,
            metadata=schema(
                title="max_depth",
                description="Maximum tree depth for base learners.",
            ),
        )

        n_estimators: XGBClassifierNEstimators = field(
            default_factory=XGBClassifierNEstimators,
            metadata=schema(
                title="n_estimators",
                description="Number of gradient boosted trees."
                "Equivalent to number of boosting rounds.",
            ),
        )

        learning_rate: XGBClassifierLearningRate = field(
            default_factory=XGBClassifierLearningRate,
            metadata=schema(
                title="learning_rate",
                description="Weight applied to each classifier"
                "at each boosting iteration. A higher learning rate"
                "increases the contribution of each classifier. "
                "There is a trade-off between the learning_rate"
                "and n_estimators parameters.",
            ),
        )

        subsample: XGBClassifierSubsample = field(
            default_factory=XGBClassifierSubsample,
            metadata=schema(
                title="subsample",
                description="Subsample ratio of the training instances."
                "Setting it to 0.5 means that XGBoost"
                " randomly samples half of the training data prior to growing trees.",
            ),
        )

        gamma: XGBClassifierGamma = field(
            default_factory=XGBClassifierGamma,
            metadata=schema(
                title="gamma",
                description="Minimum loss reduction required to make further partitions"
                " on a leaf in the tree. Set to `0` by default so no minimum loss reduction"
                " required to make a further partition on a leaf node. This means that,"
                " by default, XGBoost splits nodes as long as it improves the model at all."
                " Increasing gamma makes the algorithm more conservative by requiring a larger"
                " loss reduction before making a split, which can help control overfitting.",
            ),
        )

        colsample_bytree: XGBClassifierColSampleByTree = field(
            default_factory=XGBClassifierColSampleByTree,
            metadata=schema(
                title="Column sample by tree",
                description="Subsample ratio of columns when constructing each tree."
                "Setting it to 0.5 means that XGBoost"
                " randomly samples half of the features prior to growing trees.",
            ),
        )

    name: Literal["XGBClassifier"] = "XGBClassifier"
    parameters: Parameters = field(default_factory=Parameters)


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
        class XGBRegressorMaxDepth:
            low: int = field(default=2, metadata=schema(min=1))
            high: int = field(default=12, metadata=schema(min=1))

        @dataclass
        class XGBRegressorNEstimators:
            low: int = field(default=100, metadata=schema(min=1))
            high: int = field(default=250, metadata=schema(min=1))

        @dataclass
        class XGBRegressorLearningRate:
            low: float = field(default=0.01, metadata=schema(min=0.0001))
            high: float = field(default=3, metadata=schema(min=0.001))

        @dataclass
        class XGBRegressorSubsample:
            low: float = field(default=0.5, metadata=schema(min=0.01))
            high: float = field(default=1.0, metadata=schema(min=0.01))

        @dataclass
        class XGBRegressorGamma:
            low: float = field(default=1e-3, metadata=schema(min=1e-4))
            high: float = field(default=10, metadata=schema(min=1e-4))

        @dataclass
        class XGBRegressorColSampleByTree:
            low: float = field(default=0.6, metadata=schema(min=0.0))
            high: float = field(default=1.0, metadata=schema(min=0.0))

        max_depth: XGBRegressorMaxDepth = field(
            default_factory=XGBRegressorMaxDepth,
            metadata=schema(
                title="max_depth",
                description="Maximum tree depth for base learners.",
            ),
        )

        n_estimators: XGBRegressorNEstimators = field(
            default_factory=XGBRegressorNEstimators,
            metadata=schema(
                title="n_estimators",
                description="Number of gradient boosted trees."
                "Equivalent to number of boosting rounds.",
            ),
        )

        learning_rate: XGBRegressorLearningRate = field(
            default_factory=XGBRegressorLearningRate,
            metadata=schema(
                title="learning_rate",
                description="Weight applied to each classifier"
                "at each boosting iteration. A higher learning rate"
                "increases the contribution of each classifier. "
                "There is a trade-off between the learning_rate"
                "and n_estimators parameters.",
            ),
        )

        subsample: XGBRegressorSubsample = field(
            default_factory=XGBRegressorSubsample,
            metadata=schema(
                title="subsample",
                description="Subsample ratio of the training instances."
                "Setting it to 0.5 means that XGBoost"
                " randomly samples half of the training data prior to growing trees.",
            ),
        )

        gamma: XGBRegressorGamma = field(
            default_factory=XGBRegressorGamma,
            metadata=schema(
                title="gamma",
                description="Minimum loss reduction required to make further partitions"
                " on a leaf in the tree. Set to `0` by default so no minimum loss reduction"
                " required to make a further partition on a leaf node. This means that,"
                " by default, XGBoost splits nodes as long as it improves the model at all."
                " Increasing gamma makes the algorithm more conservative by requiring a larger"
                " loss reduction before making a split, which can help control overfitting.",
            ),
        )

        colsample_bytree: XGBRegressorColSampleByTree = field(
            default_factory=XGBRegressorColSampleByTree,
            metadata=schema(
                title="Column sample by tree",
                description="Subsample ratio of columns when constructing each tree."
                "Setting it to 0.5 means that XGBoost"
                " randomly samples half of the features prior to growing trees.",
            ),
        )

    name: Literal["XGBRegressor"] = "XGBRegressor"
    parameters: Parameters = field(default_factory=Parameters)


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

        use_py_gini: int = field(
            default=1,
            metadata=schema(
                min=0,
                max=1,
                title="Use pY GINI",
                description="The probability of y is used in GINI when this is True",
            ),
        )

        use_py_leafs: int = field(
            default=1,
            metadata=schema(
                min=0,
                max=1,
                title="Use pY leafs",
                description="The probability of y is used in leaves when this is True",
            ),
        )

        max_depth: PRFClassifierParametersMaxDepth = field(
            default_factory=PRFClassifierParametersMaxDepth,
            metadata=schema(
                title="max_depth",
                description="The maximum depth of the tree.",
            ),
        )

        n_estimators: PRFClassifierParametersNEstimators = field(
            default_factory=PRFClassifierParametersNEstimators,
            metadata=schema(
                title="n_estimators",
                description="The number of trees in the forest.",
            ),
        )

        max_features: List[PRFClassifierMaxFeatures] = field(
            default_factory=lambda: [PRFClassifierMaxFeatures.AUTO],
            metadata=schema(
                title="max_features",
                description="The number of features to consider"
                "when looking for the best split: "
                " - If “auto”, then `max_features=sqrt(n_features)`."
                " - If “sqrt”, then `max_features=sqrt(n_features)`."
                " - If “log2”, then `max_features=log2(n_features)`.",
            ),
        )

        min_py_sum_leaf: PRFClassifierParametersMinPySumLeaf = field(
            default_factory=PRFClassifierParametersMinPySumLeaf,
            metadata=schema(
                title="n_estimators",
                description="This parameter allows tree pruning when the propagation "
                "probability is small, thus reducing computation time. "
                "This value defines the probability threshold, `pth` as"
                " described in the Selective propagation scheme in the "
                "original publication `Probabilistic Random Forest: A "
                "machine learning algorithm for noisy datasets`",
            ),
        )

    name: Literal["PRFClassifier"] = "PRFClassifier"
    parameters: Parameters = field(default_factory=Parameters)


class TabPFNFeatureSelection(str, Enum):
    """Feature selection method used to reduce the number of features for TabPFNClassifier and TabPFNRegressor"""

    K_BEST = "k_best"
    "Applies SelectKBest from Scikit-Learn to select the top k features based on univariate statistical tests."
    TREE = "tree"
    "Uses feature importance from a tree-based model (Random Forest) to select features."


class TabPFNClassifierEvalMetric(str, Enum):
    """Specifies the evaluation metric used for the guided early stopping (GES) strategy during training"""

    ACCURACY = "accuracy"
    "Accuracy as the scoring metric defined the ratio of correctly predicted instances to the total instances."
    ROC_AUC = "roc_auc"
    "The area under ROC is suited for imbalanced datasets due to ranking across all possible thresholds."
    F1 = "f1"
    " F1 score capturing the harmonic mean of precision and recall, which commonly used for imbalanced datasets"
    LOG_LOSS = "log_loss"
    "Uses log loss which captures the delta between the predicted probabilities and the actual class labels."


@dataclass
class TabPFNClassifier(Algorithm):
    """TabPFN Classifier (Foundation Model for Tabular Data).

    TabPFN is a pre-trained transformer-based neural network model designed specifically for tabular data,
    allowing it to make accurate predictions on small to medium-sized datasets with minimal fine-tuning by
    training thousands of possible models on synthetic data. On a high level, it uses learned Bayesian reasoning
    to generalize from a wide variety of possible tasks. Inference requires a forward pass through a
    large transformer, so compute and memory requirements increase significantly as the number of features and
    training samples increase, which limits scalability on larger datasets.

    This implementation is based on AutoTabPFNClassifier. For optimal performance, the AutoTabPFNClassifier
    uses post-hoc ensembling which combines multiple TabPFN models and Random Forests into an ensemble. If more than
    the maximum features are present (less than 500 recommended), TabPFNClassifier will use feature selection to
    reduce the number of features to 500. Use GPU a for optimal performance.
    """

    @type_name("TabPFNClassifierParams")
    @dataclass
    class Parameters:
        max_time: int = field(
            default=150,
            metadata=schema(
                min=1,
                title="Max Time",
                description="Maximum time in seconds to train the model.",
            ),
        )

        random_state: int = field(
            default=42,
            metadata=schema(
                title="Random State",
                description="Set the seed for the TabPFNClassifier & RF algorithm",
            ),
        )

        max_feats: int = field(
            default=500,
            metadata=schema(
                min=1,
                title="Max Features",
                description="Maximum number of features allowed before feature selection is applied.",
            ),
        )

        feature_selection: List[TabPFNFeatureSelection] = field(
            default_factory=lambda: [TabPFNFeatureSelection.K_BEST],
            metadata=schema(
                title="Feature Selection",
                description="How to select features when the number of features is greater than max_feats.",
            ),
        )

        eval_metric: List[TabPFNClassifierEvalMetric] = field(
            default_factory=lambda: [TabPFNClassifierEvalMetric.ACCURACY],
            metadata=schema(
                title="Eval Metric",
                description="Specifies the training evaluation metric used for guided early stopping (GES).",
            ),
        )

    name: Literal["TabPFNClassifier"] = "TabPFNClassifier"
    parameters: Parameters = field(default_factory=Parameters)


class TabPFNRegressorEvalMetric(str, Enum):
    """Specifies the evaluation metric used for the guided early stopping (GES) strategy during training"""

    MSE = "mse"
    "Mean Squared Error (MSE) should be used to when large errors are particularly undesirable."
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    "Root Mean Squared Error (RMSE) is used to penalise large errors with more intuitive scaling."
    MAE = "mae"
    "Mean Absolute Error (MAE) is used to reduce the average magnitude of errors (without considering direction)."


@dataclass
class TabPFNRegressor(Algorithm):
    """TabPFN Regressor (Foundation Model for Tabular Data).

    TabPFN is a pre-trained transformer-based neural network model designed specifically for tabular data,
    allowing it to make accurate predictions on small to medium-sized datasets with minimal fine-tuning by
    training thousands of possible models on synthetic data. On a high level, it uses learned Bayesian reasoning
    to generalize from a wide variety of possible tasks. Inference requires a forward pass through a
    large transformer, so compute and memory requirements increase significantly as the number of features and
    training samples increase, which limits scalability on larger datasets.

    This implementation is based on AutoTabPFNRegressor. For optimal performance, the AutoTabPFNRegressor
    uses post-hoc ensembling which combines multiple TabPFN models and Random Forests into an ensemble. If more than
    the maximum features are present (less than 500 recommended), TabPFNRegressor will use feature selection to
    reduce the number of features to 500. Use GPU a for optimal performance.
    """

    @type_name("TabPFNRegressorParams")
    @dataclass
    class Parameters:
        max_time: int = field(
            default=150,
            metadata=schema(
                min=1,
                title="Max Time",
                description="Maximum time in seconds to train the model.",
            ),
        )

        random_state: int = field(
            default=42,
            metadata=schema(
                title="Random State",
                description="Set the seed for the TabPFNRegressor & RF algorithm",
            ),
        )

        max_feats: int = field(
            default=500,
            metadata=schema(
                min=1,
                title="Max Features",
                description="Maximum number of features allowed before feature selection is applied.",
            ),
        )

        feature_selection: List[TabPFNFeatureSelection] = field(
            default_factory=lambda: [TabPFNFeatureSelection.K_BEST],
            metadata=schema(
                title="Feature Selection",
                description="How to select features when the number of features is greater than max_feats.",
            ),
        )

        eval_metric: List[TabPFNRegressorEvalMetric] = field(
            default_factory=lambda: [TabPFNRegressorEvalMetric.ROOT_MEAN_SQUARED_ERROR],
            metadata=schema(
                title="Eval Metric",
                description="Specifies the training evaluation metric used for guided early stopping (GES).",
            ),
        )

    name: Literal["TabPFNRegressor"] = "TabPFNRegressor"
    parameters: Parameters = field(default_factory=Parameters)


class FastPropBatchSize(int, Enum):
    """
    Size of the batches to trial in FastProp
    """

    XSMALL = 16
    "Batches size of 16"
    SMALL = 32
    "Batches size of 32"
    MEDIUM = 64
    "Batches size of 64"
    LARGE = 256
    "Batches size of 256"
    XLARGE = 256
    "Batches size of 256"


@dataclass
class FastPropClassifier(Algorithm):
    """FastProp Classifier"""

    @type_name("FastPropClassifierParams")
    @dataclass
    class Parameters:
        @type_name("FastPropClassifierParametersHidden_Size")
        @dataclass
        class FastPropClassifierParametersHidden_Size:
            low: int = field(default=100, metadata=schema(min=100))
            high: int = field(default=3000, metadata=schema(min=100))
            step: int = field(default=100, metadata=schema(min=100))

        @type_name("FastPropClassifierParametersFNN_Layers")
        @dataclass
        class FastPropClassifierParametersFNN_Layers:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=5, metadata=schema(min=1))
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("FastPropClassifierParametersLearning_Rate")
        @dataclass
        class FastPropClassifierParametersLearning_Rate:
            low: float = field(default=1e-5, metadata=schema(min=1e-6))
            high: float = field(default=1e-2, metadata=schema(min=1e-6))

        random_seed: int = field(
            default=42,
            metadata=schema(
                title="Random Seed",
                description="Set the seed for the TabPFNClassifier & RF algorithm",
            ),
        )

        number_epochs: int = field(
            default=100,
            metadata=schema(
                min=1,
                title="Epochs",
                description="Number maximum epochs to run (increasing this and patience will increase run time)",
            ),
        )

        number_repeats: int = field(
            default=1,
            metadata=schema(
                min=1,
                title="Number of repeats",
                description="Number of repeats (ensembles) to perform (enables uncertainty but increases run time)",
            ),
        )

        patience: int = field(
            default=10,
            metadata=schema(
                min=0,
                title="Patience",
                description="Number epochs before early stopping",
            ),
        )

        train_size: float = field(
            default=0.9,
            metadata=schema(
                min=0.001,
                title="Train Size",
                description="",
            ),
        )

        val_size: float = field(
            default=0.15,
            metadata=schema(
                min=0.001,
                title="Validation Size",
                description="",
            ),
        )

        test_size: float = field(
            default=0.05,
            metadata=schema(
                min=0.001,
                title="Test Size",
                description="",
            ),
        )

        hidden_size: FastPropClassifierParametersHidden_Size = field(
            default_factory=FastPropClassifierParametersHidden_Size,
            metadata=schema(
                min=1,
                title="Hidden Size",
                description="Dimensionality of hidden layers in the FFN.",
            ),
        )

        fnn_layers: FastPropClassifierParametersFNN_Layers = field(
            default_factory=FastPropClassifierParametersFNN_Layers,
            metadata=schema(
                min=1,
                title="FNN Layers",
                description="Number of layers in the neural network.",
            ),
        )

        learning_rate: FastPropClassifierParametersLearning_Rate = field(
            default_factory=FastPropClassifierParametersLearning_Rate,
            metadata=schema(
                title="Learning Rate",
                description="The learning rate",
            ),
        )

        batch_size: List[FastPropBatchSize] = field(
            default_factory=lambda: [FastPropBatchSize.MEDIUM],
            metadata=schema(
                title="Batch Size",
                description="How many samples per batch to load.",
            ),
        )

    name: Literal["FastPropClassifier"] = "FastPropClassifier"
    parameters: Parameters = field(default_factory=Parameters)


@dataclass
class FastPropRegressor(Algorithm):
    """FastProp Regressor"""

    @type_name("FastPropRegressorParams")
    @dataclass
    class Parameters:
        @type_name("FastPropRegressorParametersHidden_Size")
        @dataclass
        class FastPropRegressorParametersHidden_Size:
            low: int = field(default=100, metadata=schema(min=100))
            high: int = field(default=3000, metadata=schema(min=100))
            step: int = field(default=100, metadata=schema(min=100))

        @type_name("FastPropRegressorParametersFNN_Layers")
        @dataclass
        class FastPropRegressorParametersFNN_Layers:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=5, metadata=schema(min=1))
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("FastPropRegressorParametersLearning_Rate")
        @dataclass
        class FastPropRegressorParametersLearning_Rate:
            low: float = field(default=1e-5, metadata=schema(min=1e-6))
            high: float = field(default=1e-2, metadata=schema(min=1e-6))

        random_seed: int = field(
            default=42,
            metadata=schema(
                title="Random Seed",
                description="Set the seed for the TabPFNClassifier & RF algorithm",
            ),
        )

        number_epochs: int = field(
            default=100,
            metadata=schema(
                min=1,
                title="Epochs",
                description="Number maximum epochs to run (increasing this and patience will increase run time)",
            ),
        )

        number_repeats: int = field(
            default=1,
            metadata=schema(
                min=1,
                title="Number of repeats",
                description="Number of repeats (ensembles) to perform (enables uncertainty but increases run time)",
            ),
        )

        patience: int = field(
            default=10,
            metadata=schema(
                min=0,
                title="Patience",
                description="Number epochs before early stopping",
            ),
        )

        train_size: float = field(
            default=0.9,
            metadata=schema(
                min=0.001,
                title="Train Size",
                description="",
            ),
        )

        val_size: float = field(
            default=0.15,
            metadata=schema(
                min=0.001,
                title="Validation Size",
                description="",
            ),
        )

        test_size: float = field(
            default=0.05,
            metadata=schema(
                min=0.001,
                title="Test Size",
                description="",
            ),
        )

        hidden_size: FastPropRegressorParametersHidden_Size = field(
            default_factory=FastPropRegressorParametersHidden_Size,
            metadata=schema(
                min=1,
                title="Hidden Size",
                description="Dimensionality of hidden layers in the FFN.",
            ),
        )

        fnn_layers: FastPropRegressorParametersFNN_Layers = field(
            default_factory=FastPropRegressorParametersFNN_Layers,
            metadata=schema(
                min=1,
                title="FNN Layers",
                description="Number of layers in the neural network.",
            ),
        )

        learning_rate: FastPropRegressorParametersLearning_Rate = field(
            default_factory=FastPropRegressorParametersLearning_Rate,
            metadata=schema(
                title="Learning Rate",
                description="The learning rate",
            ),
        )

        batch_size: List[FastPropBatchSize] = field(
            default_factory=lambda: [FastPropBatchSize.MEDIUM],
            metadata=schema(
                title="Batch Size",
                description="How many samples per batch to load.",
            ),
        )

    name: Literal["FastPropRegressor"] = "FastPropRegressor"
    parameters: Parameters = field(default_factory=Parameters)


class ChemPropActivation(str, Enum):
    """
    The activation function to use within the network.
    See https://chemprop.readthedocs.io/en/latest/args.html#chemprop.args.TrainArgs.activation for details
    """

    RELU = "RELU"
    TANH = "TANH"
    LEAKYRELU = "LEAKYRELU"
    PRELU = "PRELU"
    SELU = "SELU"
    ELU = "ELU"


class ChemPropMolecule_Featurizers(str, Enum):
    """
    Features generators are used for computing additional molecule-level features that are appended
    after message passing.
    See https://chemprop.readthedocs.io/en/latest/extra_features_descriptors.html#Extra-molecule-features
    for details.
    """

    MORGAN = "morgan_binary"
    "Generates a binary Morgan fingerprint for a molecule."
    MORGAN_COUNT = "morgan_count"
    "Generates a counts-based Morgan fingerprint for a molecule."
    RDKIT_2D = "rdkit_2d"
    "Generates RDKit 2D features for a molecule."
    V1_RDKIT_2D_NORMALIZED = "v1_rdkit_2d_normalized"
    "Generates version 1 RDKit 2D normalized features for a molecule."
    AVALON = "avalon"
    "Qptuna implementation of Avalon."
    UNSCALEDJAZZYDESCRIPTORS = "unscaledjazzydescriptors"
    "Qptuna implementation of Jazzy."
    UNSCALEDPHYSCHEMDESCRIPTORS = "unscaledphyschemdescriptors"
    "Qptuna implementation of Physchem."
    UNSCALEDMAPC = "unscaledmapc"
    "Qptuna implementation of MAPC."
    MORDREDDESCRIPTORS = "mordreddescriptors"
    "Qptuna implementation of Mordred."


class ChemPropBatchSize(int, Enum):
    """
    Features generators are used for computing additional molecule-level features that are appended
    after message passing.
    See https://chemprop.readthedocs.io/en/latest/features.html#features-generators for details.
    """

    XSMALL = 16
    "Batches size of 16"
    SMALL = 32
    "Batches size of 32"
    MEDIUM = 64
    "Batches size of 64"
    LARGE = 128
    "Batches size of 128"
    XLARGE = 256
    "Batches size of 256"


class ChemPropAggregation(str, Enum):
    """Atom-level representations from the MPNN"""

    MEAN = "mean"
    "Representations averaged over all atoms of a molecule"
    SUM = "sum"
    "Representations summed over all atoms of a molecule"
    NORM = "norm"
    "Representations summed up and divided by a constant (default=100)"


class ChemPropTrialDropout(IntEnum):
    """Whether to trial dropout (note this enables trials to evaluate dropout from a probability). It does not activate
    dropout in every trial)"""

    YES = 1
    "Trial dropout with recommended trails"
    NO = 0
    "Do not perform dropout in any trials"


class ChemPropUndirected(IntEnum):
    """Whether to trial undirected"""

    YES = 1
    "Trial undirected within trails"
    NO = 0
    "Do not use undirected in any trials"


class ChemPropBatch_Norm(IntEnum):
    """Whether to trial undirected"""

    YES = 1
    "Trial batch_norm within trails"
    NO = 0
    "Do not use batch_norm in any trials"


class ChemPropMessage_Bias(IntEnum):
    """Whether to trial undirected"""

    YES = 1
    "Trial message bias within trails"
    NO = 0
    "Do not use message bias in any trials"


class ChemPropClassifierLossFunction(str, Enum):
    """Loss functions for the ChemPropClassifier. This is used to define the loss which is minimised during training."""

    BCE = "bce"
    "Binary cross-entropy (default)"
    BINARYMCC = "binary-mcc"
    "Binary Matthews correlation coefficient"
    DIRICHLET = "dirichlet"
    "Representations summed up and divided by a constant (default=100)"


@dataclass
class ChemPropClassifier(Algorithm):
    """Chemprop Classifier

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
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropClassifierDepth")
        @dataclass
        class ChemPropParametersDepth:
            low: int = field(default=2, metadata=schema(min=2))
            high: int = field(default=6, metadata=schema(min=2))
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropClassifierFFN_Hidden_Dim")
        @dataclass
        class ChemPropParametersFFN_Hidden_Dim:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            step: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropClassifierFFN_Num_Layers")
        @dataclass
        class ChemPropParametersFFN_Num_Layers:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=3, metadata=schema(min=1))
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropClassifierFinal_Lr_Ratio")
        @dataclass
        class ChemPropParametersFinal_Lr_Ratio:
            low: float = field(default=1e-2, metadata=schema(min=1e-6))
            high: float = field(default=1, metadata=schema(min=1e-6))

        @type_name("ChemPropClassifierMessage_Hidden_Dim")
        @dataclass
        class ChemPropParametersMessage_Hidden_Dim:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            step: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropClassifierInit_Lr_Ratio")
        @dataclass
        class ChemPropParametersInit_Lr_Ratio:
            low: float = field(default=1e-2, metadata=schema(min=1e-6))
            high: float = field(default=1, metadata=schema(min=1e-6))

        @type_name("ChemPropClassifierMax_Lr")
        @dataclass
        class ChemPropParametersMax_Lr:
            low: float = field(default=1e-4, metadata=schema(min=1e-6))
            high: float = field(default=1e-2, metadata=schema(min=1e-6))

        @type_name("ChemPropClassifierWarmup_Epochs_Ratio")
        @dataclass
        class ChemPropParametersWarmup_Epochs_Ratio:
            low: float = field(default=0.1, metadata=schema(min=0.0))
            high: float = field(default=0.1, metadata=schema(min=0.0))
            step: float = field(default=0.1, metadata=schema(min=0.0))

        batch_norm: List[ChemPropBatch_Norm] = field(
            default_factory=lambda: [ChemPropBatch_Norm.NO],
            metadata=schema(
                title="Batch normalisation",
                description="If True, apply batch normalization to the output of the aggregation operation",
            ),
        )

        ensemble_size: int = field(
            default=1,
            metadata=schema(
                min=1,
                title="Ensemble size",
                description="Number of ensembles with different weight initialisation (provides uncertainty)",
            ),
        )

        epochs: int = field(
            default=100,
            metadata=schema(
                min=1,
                title="Epochs",
                description="Number maximum epochs to run (increasing this and patience will increase run time)",
            ),
        )

        patience: int = field(
            default=10,
            metadata=schema(
                min=0,
                title="Patience",
                description="Number epochs before early stopping",
            ),
        )

        activation: List[ChemPropActivation] = field(
            default_factory=lambda: [ChemPropActivation.RELU],
            metadata=schema(
                title="activation",
                description="Activation function applied to the "
                "output of the weighted sum of inputs",
            ),
        )

        aggregation: List[ChemPropAggregation] = field(
            default_factory=lambda: [ChemPropAggregation.NORM],
            metadata=schema(
                title="aggregation",
                description="Aggregation scheme for atomic vectors into molecular vectors.",
            ),
        )

        aggregation_norm: ChemPropParametersAggregation_Norm = field(
            default_factory=ChemPropParametersAggregation_Norm,
            metadata=schema(
                min=1,
                title="aggregation_norm",
                description="For norm aggregation, number by which to divide summed up atomic features.",
            ),
        )

        batch_size: List[ChemPropBatchSize] = field(
            default_factory=lambda: [ChemPropBatchSize.MEDIUM],
            metadata=schema(
                title="batch_size",
                description="How many samples per batch to load.",
            ),
        )

        depth: ChemPropParametersDepth = field(
            default_factory=ChemPropParametersDepth,
            metadata=schema(
                min=1,
                title="depth",
                description="Number of message passing steps"
                "(distance of neighboring atoms visible when modelling).",
            ),
        )

        trial_dropout: ChemPropTrialDropout = field(
            default=ChemPropTrialDropout.NO,
            metadata=schema(
                title="Trial dropout",
                description="Whether to trial dropout. During training, randomly zeroes"
                " some of the elements of the input tensor with probability `p`"
                " using samples from a Bernoulli distribution. Each channel will"
                " be zeroed out independently on every forward call.  This has"
                " proven to be an effective technique for regularization and "
                "preventing the co-adaptation of neurons",
            ),
        )

        loss_function: List[ChemPropClassifierLossFunction] = field(
            default_factory=lambda: [ChemPropClassifierLossFunction.BCE],
            metadata=schema(
                title="Loss function",
                description="Defines the function used to describe the loss of the MPNN classifier during training.",
            ),
        )

        molecule_featurizers: List[ChemPropMolecule_Featurizers] | None = field(
            default=None,
            metadata=schema(
                title="molecule_featurizers",
                description="Method of generating additional features.",
            ),
        )

        message_bias: List[ChemPropMessage_Bias] = field(
            default_factory=lambda: [ChemPropMessage_Bias.NO],
            metadata=schema(
                title="Message bias",
                description="Add bias to the message passing layers, which might improve performance.",
            ),
        )

        ffn_hidden_dim: ChemPropParametersFFN_Hidden_Dim = field(
            default_factory=ChemPropParametersFFN_Hidden_Dim,
            metadata=schema(
                min=1,
                title="ffn_hidden_dim",
                description="Dimensionality of hidden layers in the FFN.",
            ),
        )

        ffn_num_layers: ChemPropParametersFFN_Num_Layers = field(
            default_factory=ChemPropParametersFFN_Num_Layers,
            metadata=schema(
                min=1,
                title="ffn_num_layers",
                description="Number of layers in the FFN after D-MPNN encoding.",
            ),
        )

        final_lr_ratio: ChemPropParametersFinal_Lr_Ratio = field(
            default_factory=ChemPropParametersFinal_Lr_Ratio,
            metadata=schema(
                title="final_lr_ratio",
                description="The final learning rate ratio.",
            ),
        )

        message_hidden_dim: ChemPropParametersMessage_Hidden_Dim = field(
            default_factory=ChemPropParametersMessage_Hidden_Dim,
            metadata=schema(
                min=1,
                title="message_hidden_dim",
                description="Size of the hidden bond message vectors in the D-MPNN",
            ),
        )

        init_lr_ratio: ChemPropParametersInit_Lr_Ratio = field(
            default_factory=ChemPropParametersInit_Lr_Ratio,
            metadata=schema(
                title="init_lr_ratio",
                description="The learning rate ratio.",
            ),
        )

        max_lr: ChemPropParametersMax_Lr = field(
            default_factory=ChemPropParametersMax_Lr,
            metadata=schema(
                title="max_lr",
                description="The maximum learning rate.",
            ),
        )

        warmup_epochs_ratio: ChemPropParametersWarmup_Epochs_Ratio = field(
            default_factory=ChemPropParametersWarmup_Epochs_Ratio,
            metadata=schema(
                title="warmup_epochs_ratio",
                description="Ratio for the number of epochs during which "
                "learning rate increases linearly from init_lr to max_lr."
                " Afterwards, learning rate decreases exponentially "
                "from max_lr to final_lr.",
            ),
        )

        undirected: List[ChemPropUndirected] = field(
            default_factory=lambda: [ChemPropUndirected.NO],
            metadata=schema(
                title="Undirected",
                description="Pass messages on undirected bonds/edges (always sum the two relevant bond vectors).",
            ),
        )

    name: Literal["ChemPropClassifier"] = "ChemPropClassifier"
    parameters: Parameters = field(default_factory=Parameters)


class ChemPropRegressorLossFunction(str, Enum):
    """Loss functions for the ChemPropClassifier. This is used to define the loss which is minimised during training."""

    MSE = "mse"
    "Mean squared error (default)"
    BOUNDEDMSE = "bounded-mse"
    "Bounded mean squared error"
    MVE = "mve"
    "Mean-variance estimation (MVE) - estimates the mean and variance of the target variable. It is used to model uncertainty in regression tasks, allowing the model to predict both the expected value and the uncertainty of the prediction."
    EVIDENTIAL = "evidential"
    "Evidential regression - a probabilistic approach that models the uncertainty in predictions by estimating the parameters of a distribution over the target variable. It is particularly useful for tasks where uncertainty quantification is important."


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
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropRegressorDepth")
        @dataclass
        class ChemPropParametersDepth:
            low: int = field(default=2, metadata=schema(min=2))
            high: int = field(default=6, metadata=schema(min=2))
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropRegressorFFN_Hidden_Dim")
        @dataclass
        class ChemPropParametersFFN_Hidden_Dim:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            step: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropRegressorFFN_Num_Layers")
        @dataclass
        class ChemPropParametersFFN_Num_Layers:
            low: int = field(default=1, metadata=schema(min=1))
            high: int = field(default=3, metadata=schema(min=1))
            step: int = field(default=1, metadata=schema(min=1))

        @type_name("ChemPropRegressorFinal_Lr_Ratio")
        @dataclass
        class ChemPropParametersFinal_Lr_Ratio:
            low: float = field(default=1e-2, metadata=schema(min=1e-6))
            high: float = field(default=1, metadata=schema(min=1e-6))

        @type_name("ChemPropRegressorMessage_Hidden_Dim")
        @dataclass
        class ChemPropParametersMessage_Hidden_Dim:
            low: int = field(default=300, metadata=schema(min=300))
            high: int = field(default=2400, metadata=schema(min=300))
            step: int = field(default=100, metadata=schema(min=100))

        @type_name("ChemPropRegressorInit_Lr_Ratio")
        @dataclass
        class ChemPropParametersInit_Lr_Ratio:
            low: float = field(default=1e-2, metadata=schema(min=1e-6))
            high: float = field(default=1, metadata=schema(min=1e-6))

        @type_name("ChemPropRegressorMax_Lr")
        @dataclass
        class ChemPropParametersMax_Lr:
            low: float = field(default=1e-4, metadata=schema(min=1e-6))
            high: float = field(default=1e-2, metadata=schema(min=1e-6))

        @type_name("ChemPropRegressorWarmup_Epochs_Ratio")
        @dataclass
        class ChemPropParametersWarmup_Epochs_Ratio:
            low: float = field(default=0.1, metadata=schema(min=0.0))
            high: float = field(default=0.1, metadata=schema(min=0.0))
            step: float = field(default=0.1, metadata=schema(min=0.0))

        batch_norm: List[ChemPropBatch_Norm] = field(
            default_factory=lambda: [ChemPropBatch_Norm.NO],
            metadata=schema(
                title="Batch normalisation",
                description="If True, apply batch normalization to the output of the aggregation operation",
            ),
        )

        ensemble_size: int = field(
            default=1,
            metadata=schema(
                min=1,
                title="Ensemble size",
                description="Number of ensembles with different weight initialisation (provides uncertainty)",
            ),
        )

        epochs: int = field(
            default=100,
            metadata=schema(
                min=1,
                title="Epochs",
                description="Number maximum epochs to run (increasing this and patience will increase run time)",
            ),
        )

        patience: int = field(
            default=10,
            metadata=schema(
                min=0,
                title="Patience",
                description="Number epochs before early stopping",
            ),
        )

        activation: List[ChemPropActivation] = field(
            default_factory=lambda: [ChemPropActivation.RELU],
            metadata=schema(
                title="activation",
                description="Activation function applied to the "
                "output of the weighted sum of inputs",
            ),
        )

        aggregation: List[ChemPropAggregation] = field(
            default_factory=lambda: [ChemPropAggregation.NORM],
            metadata=schema(
                title="aggregation",
                description="Aggregation scheme for atomic vectors into molecular vectors.",
            ),
        )

        aggregation_norm: ChemPropParametersAggregation_Norm = field(
            default_factory=ChemPropParametersAggregation_Norm,
            metadata=schema(
                title="aggregation_norm",
                description="For norm aggregation, number by which to divide summed up atomic features.",
            ),
        )

        batch_size: List[ChemPropBatchSize] = field(
            default_factory=lambda: [ChemPropBatchSize.MEDIUM],
            metadata=schema(
                title="batch_size",
                description="How many samples per batch to load.",
            ),
        )

        depth: ChemPropParametersDepth = field(
            default_factory=ChemPropParametersDepth,
            metadata=schema(
                title="depth",
                description="Number of message passing steps"
                "(distance of neighboring atoms visible when modelling).",
            ),
        )

        trial_dropout: ChemPropTrialDropout = field(
            default=ChemPropTrialDropout.NO,
            metadata=schema(
                title="Trial dropout",
                description="Whether to trial dropout. During training, randomly zeroes"
                " some of the elements of the input tensor with probability `p`"
                " using samples from a Bernoulli distribution. Each channel will"
                " be zeroed out independently on every forward call.  This has"
                " proven to be an effective technique for regularization and "
                "preventing the co-adaptation of neurons",
            ),
        )

        loss_function: List[ChemPropRegressorLossFunction] = field(
            default_factory=lambda: [ChemPropRegressorLossFunction.MSE],
            metadata=schema(
                title="Loss function",
                description="Defines the function used to describe the loss of the MPNN classifier during training.",
            ),
        )

        molecule_featurizers: List[ChemPropMolecule_Featurizers] | None = field(
            default=None,
            metadata=schema(
                title="molecule_featurizers",
                description="Method of generating additional features.",
            )
            | none_as_undefined,
        )

        message_bias: List[ChemPropMessage_Bias] = field(
            default_factory=lambda: [ChemPropMessage_Bias.NO],
            metadata=schema(
                title="Message bias",
                description="Add bias to the message passing layers, which might improve performance.",
            ),
        )

        ffn_hidden_dim: ChemPropParametersFFN_Hidden_Dim = field(
            default_factory=ChemPropParametersFFN_Hidden_Dim,
            metadata=schema(
                title="ffn_hidden_dim",
                description="Dimensionality of hidden layers in the FFN.",
            ),
        )

        ffn_num_layers: ChemPropParametersFFN_Num_Layers = field(
            default_factory=ChemPropParametersFFN_Num_Layers,
            metadata=schema(
                title="ffn_num_layers",
                description="Number of layers in the FFN after D-MPNN encoding.",
            ),
        )

        final_lr_ratio: ChemPropParametersFinal_Lr_Ratio = field(
            default_factory=ChemPropParametersFinal_Lr_Ratio,
            metadata=schema(
                title="final_lr_ratio",
                description="The final learning rate ratio.",
            ),
        )

        message_hidden_dim: ChemPropParametersMessage_Hidden_Dim = field(
            default_factory=ChemPropParametersMessage_Hidden_Dim,
            metadata=schema(
                title="message_hidden_dim",
                description="Size of the hidden bond message vectors in the D-MPNN",
            ),
        )

        init_lr_ratio: ChemPropParametersInit_Lr_Ratio = field(
            default_factory=ChemPropParametersInit_Lr_Ratio,
            metadata=schema(
                title="init_lr_ratio",
                description="The learning rate ratio.",
            ),
        )

        max_lr: ChemPropParametersMax_Lr = field(
            default_factory=ChemPropParametersMax_Lr,
            metadata=schema(
                title="max_lr",
                description="The maximum learning rate.",
            ),
        )

        warmup_epochs_ratio: ChemPropParametersWarmup_Epochs_Ratio = field(
            default_factory=ChemPropParametersWarmup_Epochs_Ratio,
            metadata=schema(
                title="warmup_epochs_ratio",
                description="Ratio for the number of epochs during which "
                "learning rate increases linearly from init_lr to max_lr."
                " Afterwards, learning rate decreases exponentially "
                "from max_lr to final_lr.",
            ),
        )

        undirected: List[ChemPropUndirected] = field(
            default_factory=lambda: [ChemPropUndirected.NO],
            metadata=schema(
                title="Undirected",
                description="Pass messages on undirected bonds/edges (always sum the two relevant bond vectors).",
            ),
        )

    name: Literal["ChemPropRegressor"] = "ChemPropRegressor"
    parameters: Parameters = field(default_factory=Parameters)


class ChemPropFrzn(str, Enum):
    """
    `Qptuna` implements a hyperparameter search space level for ChemProp in order to define Hyperopt search space
     to optimise. Increasing levels correspond to increasing the search space.
    """

    MPNN = "mpnn"
    """Freeze the weights in only the MPNN during transfer learning"""
    MPNN_FIRST_FFN = "mpnn_first_ffn"
    """Freeze the MPNN and first layer of the FFN during transfer learning"""
    MPNN_LAST_FFN = "mpnn_last_ffn"
    """Freeze the MPNN and until the penultimate layer of the FFN during transfer learning"""
    FIRST_FFN = "first_ffn"
    """Freeze the first layer of the FFN during transfer learning"""
    LAST_FFN = "last_ffn"
    """Freeze the FFN until the penultimate layer during transfer learning"""


@dataclass
class ChemPropRegressorPretrained(Algorithm):
    """Chemprop Regressor from a pretrined model

    Pretraining can be carried out by supplying previously trained Qptuna ChemProp PKL model.
    """

    @type_name("ChemPropRegressorPretrainedParams")
    @dataclass
    class Parameters:
        epochs: int = field(
            default=100,
            metadata=schema(
                min=0,
                title="Epochs",
                description="Number maximum epochs to run (increasing this and patience will increase run time)",
            ),
        )

        patience: int = field(
            default=10,
            metadata=schema(
                min=0,
                title="Patience",
                description="Number epochs before early stopping",
            ),
        )

        frzn: List[ChemPropFrzn] | None = field(
            default=None,
            metadata=schema(
                title="Frozen layers",
                description="Decide if layers of the MPNN or FFN to freeze during transfer learning.",
            ),
        )

        pretrained_model: str = field(
            default=None,
            metadata=schema(
                title="Pretrained Model",
                description="Path to a pretrained Qptuna pkl model",
            ),
        )

    name: Literal["ChemPropRegressorPretrained"] = "ChemPropRegressorPretrained"
    parameters: Parameters = field(default_factory=Parameters)


@dataclass
class CustomClassificationModel(Algorithm):
    """Classifier from a preexisting pkl model

    CustomClassificationModel is used to load Qptuna or scikit-learn (like) classifier models into Qptuna. Models do not
    require previous fitting, but they must follow the scikit-learn API schema for a classifier.

    This method requires a path to a preexisting model and a `refit` parameter used to determine whether to update
    the model with new training data and retrain the custom algorithm. If refitting is disabled, no retraining is
    performed ( performance reflects your previous model fitted with prior data), enabling easy evaluation of
    historical models to newer data. If refitting is enabled then your algorithm is fit to the new train data (which
    will likely overwrite your previous fitting, unless your algorithm is configured otherwise), enabling easy
    evaluation of unfitted custom algorithms or reuse of specific hyperparameters.
    """

    @type_name("CustomClassificationModelParams")
    @dataclass
    class Parameters:
        model_file: str = field(
            default=None,
            metadata=schema(
                title="Preexisting Model File",
                description="Path to a file of a preexisting pkl model",
            ),
        )
        refit_model: int = field(
            default=0,
            metadata=schema(
                min=0,
                max=1,
                title="Refit Model",
                description="Whether fit should be called during training",
            ),
        )

    name: Literal["CustomClassificationModel"] = "CustomClassificationModel"
    parameters: Parameters = field(default_factory=Parameters)


@dataclass
class CustomRegressionModel(Algorithm):
    """Regressor from a preexisting pkl model

    CustomRegressionModel is used to load Qptuna or scikit-learn (like) regressor models into Qptuna. Models do not
    require previous fitting, but they must follow the scikit-learn API schema for a regressor.

    This method requires a path to a preexisting model and a `refit` parameter used to determine whether to update
    the model with new training data and retrain the custom algorithm. If refitting is disabled, no retraining is
    performed ( performance reflects your previous model fitted with prior data), enabling easy evaluation of
    historical models to newer data. If refitting is enabled then your algorithm is fit to the new train data (which
    will likely overwrite your previous fitting, unless your algorithm is configured otherwise), enabling easy
    evaluation of unfitted custom algorithms or reuse of specific hyperparameters.
    """

    @type_name("CustomRegressionModelParams")
    @dataclass
    class Parameters:
        model_file: str = field(
            default=None,
            metadata=schema(
                title="Preexisting Model File",
                description="Path to a file of a preexisting pkl model",
            ),
        )
        refit_model: int = field(
            default=0,
            metadata=schema(
                min=0,
                max=1,
                title="Refit Model",
                description="Whether the fit function should be called during training",
            ),
        )

    name: Literal["CustomRegressionModel"] = "CustomRegressionModel"
    parameters: Parameters = field(default_factory=Parameters)


AnyRegressionAlgorithm = Union[
    Lasso,
    CatBoostRegressor,
    PLSRegression,
    RandomForestRegressor,
    Ridge,
    KNeighborsRegressor,
    SVR,
    XGBRegressor,
    PRFClassifier,  # PRFClassifier ingests/outputs continuous probabilities so should be evaluated as regressor
    TabPFNRegressor,
    FastPropRegressor,
    ChemPropRegressor,
    ChemPropRegressorPretrained,
    CustomRegressionModel,
]

AnyClassificationAlgorithm = Union[
    AdaBoostClassifier,
    CatBoostClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    XGBClassifier,
    TabPFNClassifier,
    FastPropClassifier,
    ChemPropClassifier,
    CustomClassificationModel,
]


class CalibratedClassifierCVEnsemble(str, Enum):
    "Whether to use ensemble or not during calibration aggregation"
    TRUE = "True"
    FALSE = "False"
    # We use the string representation of the boolean to avoid serialization issues


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
        estimator: AnyClassificationAlgorithm = field(
            default=AnyClassificationAlgorithm,
            metadata=schema(
                title="Estimator",
                description="Base estimator to use for calibration",
            ),
        )

        ensemble: Union[CalibratedClassifierCVEnsemble, str] = field(
            default=CalibratedClassifierCVEnsemble.TRUE,
            metadata=schema(
                title="ensemble",
                description="Whether each cv it fits a copy of the base estimator, vs. cv used to obtain unbiased "
                "predictions used for calibration",
            ),
        )

        method: Union[CalibratedClassifierCVMethod, str] = field(
            default=CalibratedClassifierCVMethod.ISOTONIC,
            metadata=schema(
                title="method",
                description="Calibration method used to obtained calibrated predictions",
            ),
        )
        n_folds: int = field(
            default=5,
            metadata=schema(
                min=2,
                max=5,
                title="Number of Cross validation folds (splits)",
                description="Number of cv folds to obtain calibration data",
            ),
        )

    name: Literal["CalibratedClassifierCVWithVA"] = "CalibratedClassifierCVWithVA"
    parameters: Parameters = field(default_factory=Parameters)


AnyClassificationAlgorithm = Union[
    AnyClassificationAlgorithm, CalibratedClassifierCVWithVA
]

MapieClassifierCompatibleAlgorithm = Union[
    AnyClassificationAlgorithm,
    CalibratedClassifierCVWithVA,
]

MapieRegressorCompatibleAlgorithm = Union[AnyRegressionAlgorithm]


@dataclass
class MapieClassifier(Algorithm):
    """Mapie Classifier

    MAPIE - Model Agnostic Prediction Interval Estimator

    MAPIE allows you to estimate prediction intervals for regression models. Prediction intervals output by MAPIE
    encompass both aleatoric and epistemic uncertainties and are backed by strong theoretical guarantees thanks to
    conformal prediction methods.
    """

    @type_name("MapieClassifierParams")
    @dataclass
    class Parameters:
        estimator: MapieClassifierCompatibleAlgorithm = field(
            default=MapieClassifierCompatibleAlgorithm,
            metadata=schema(
                title="Estimator",
                description="Base estimator to use",
            ),
        )
        mapie_alpha: float = field(
            default=0.05,
            metadata=schema(
                title="Uncertainty alpha",
                min=0.01,
                description="Alpha used to generate uncertainty estimates",
            ),
        )
        test_size: float = field(
            default=0.10,
            metadata=schema(
                title="Test Size",
                min=0.01,
                description="Size of the test set used in calibration",
            ),
        )
        n_folds: int = field(
            default=5,
            metadata=schema(
                min=1,
                title="Number of Cross validation folds (splits)",
                description="Number of cv folds to obtain calibration data",
            ),
        )
        random_state: int = field(
            default=42,
            metadata=schema(
                title="Random State",
                description="Set the seed for the MAPIE algorithm, CV split & base algorithm",
            ),
        )

    name: Literal["MapieClassifier"] = "MapieClassifier"
    parameters: Parameters = field(default_factory=Parameters)


@dataclass
class MapieRegressor(Algorithm):
    """Mapie Regressor

    MAPIE - Model Agnostic Prediction Interval Estimator

    MAPIE allows you to estimate prediction intervals for regression models. Prediction intervals output by MAPIE
    encompass both aleatoric and epistemic uncertainties and are backed by strong theoretical guarantees thanks to
    conformal prediction methods.
    """

    @type_name("MapieRegressorParams")
    @dataclass
    class Parameters:
        estimator: MapieRegressorCompatibleAlgorithm = field(
            default=MapieRegressorCompatibleAlgorithm,
            metadata=schema(
                title="Estimator",
                description="Base estimator to use",
            ),
        )
        mapie_alpha: float = field(
            default=0.05,
            metadata=schema(
                title="Uncertainty alpha",
                min=0.01,
                description="Alpha used to generate uncertainty estimates",
            ),
        )
        test_size: float = field(
            default=0.10,
            metadata=schema(
                title="Test Size",
                min=0.01,
                description="Size of the test set used in calibration",
            ),
        )
        n_folds: int = field(
            default=5,
            metadata=schema(
                min=2,
                max=5,
                title="Number of Cross validation folds (splits)",
                description="Number of cv folds to obtain calibration data",
            ),
        )
        random_state: int = field(
            default=42,
            metadata=schema(
                title="Random State",
                description="Set the seed for the MAPIE algorithm, CV split & base algorithm",
            ),
        )

    name: Literal["MapieRegressor"] = "MapieRegressor"
    parameters: Parameters = field(default_factory=Parameters)


AnyAlgorithm = Union[
    AnyRegressionAlgorithm,
    AnyClassificationAlgorithm,
    CalibratedClassifierCVWithVA,
    MapieRegressor,
    MapieClassifier,
]

AnyChemPropAlgorithm = [
    ChemPropClassifier,
    ChemPropRegressor,
    ChemPropRegressorPretrained,
]


def detect_mode_from_algs(algs: List[AnyAlgorithm]) -> ModelMode:
    if all(isinstance(alg, AnyRegressionAlgorithm) for alg in algs):
        mode = ModelMode.REGRESSION
    elif all(isinstance(alg, AnyClassificationAlgorithm) for alg in algs):
        mode = ModelMode.CLASSIFICATION
    else:
        raise ValueError(
            f"Provided algorithms ({algs}) "
            f"are neither only regression ({AnyRegressionAlgorithm.__args__}),",
            f"nor only classification ({AnyClassificationAlgorithm.__args__})",
        )
    return mode


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

        optimization_split_strategy: AnyCvSplitter = field(
            default_factory=Stratified,
            metadata=schema(
                title="Hyperparameter optimization split strategy",
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

        n_splits: int = field(
            default=5,
            metadata=schema(
                min=1,
                title="Hyperparameter split size",
                description="Number of cross-validated splits Optuna applies during hyperparameter optimization"
                " Use '1' to disable cross-validation, but note that KFold requires at least two splits/folds. "
                "Also note only one split is used, which is repeated n_replicates times across different seeds.",
            ),
        )

        n_replicates: Optional[int] = field(
            default=5,
            metadata=schema(
                min=1,
                title="Hyperparameter replicates",
                description="Number of differently initialised cross-validators Optuna uses to calculate hyperparameter "
                "performance. Optuna utilised only the first split per cross-validation replicate and "
                "reports the mean across the n_replicates.",
            ),
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

        random_seed: int = field(
            default=42,
            metadata=schema(
                title="Seed for reproducibility",
                description="This seed is used for a random number generator."
                " Set to an integer value to get reproducible results."
                " Set to None/null/empty to initialize at random.",
            ),
        )

        leave_out: Optional[float] = field(
            default=None,
            metadata=schema(
                title="Proportion of Optimisation datapoints to leave out",
                description="Use this to reduce compute time for very large training"
                "set inputs, by leaving out a proportion of data during optimization.",
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
        if isinstance(self.settings.scoring, Enum):
            self.settings.scoring = self.settings.scoring.value

        # Set default response type.
        if self.mode == ModelMode.REGRESSION:
            self.data.response_type = "regression"
        elif self.mode == ModelMode.CLASSIFICATION:
            self.data.response_type = "classification"
