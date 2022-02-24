import abc
from dataclasses import dataclass, field
from typing import Optional, Union

import sklearn
import sklearn.cross_decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import xgboost
from apischema import schema
from sklearn.base import BaseEstimator
from typing_extensions import Literal

from optunaz.config import (
    ModelMode,
    OptimizationDirection,
    Algorithm as GenericAlg,
    Visualization,
)
from optunaz.config.optconfig import RegressionScore, ClassificationScore
from optunaz.datareader import Dataset
from optunaz.descriptors import AnyDescriptor


class Algorithm(GenericAlg):
    @abc.abstractmethod
    def estimator(self) -> BaseEstimator:
        pass


@dataclass
class AdaBoostClassifier(Algorithm):
    @dataclass
    class AdaBoostClassifierParameters:
        n_estimators: int = field(metadata=schema(min=1))
        learning_rate: float = field(metadata=schema(min=0.0001))

    name: Literal["AdaBoostClassifier"]
    parameters: AdaBoostClassifierParameters

    def estimator(self):
        return sklearn.ensemble.AdaBoostClassifier(
            base_estimator=None,
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            algorithm="SAMME.R",
        )


@dataclass
class Lasso(Algorithm):
    @dataclass
    class LassoParameters:
        alpha: float = field(default=1.0, metadata=schema(min=0))

    name: Literal["Lasso"]
    parameters: LassoParameters

    def estimator(self):
        return sklearn.linear_model.Lasso(alpha=self.parameters.alpha)


@dataclass
class LogisticRegression(Algorithm):
    @dataclass
    class LogisticRegressionParameters:
        solver: str
        C: float = field(metadata=schema(min=0.001, max=1000))

    name: Literal["LogisticRegression"]
    parameters: LogisticRegressionParameters

    def estimator(self):
        return sklearn.linear_model.LogisticRegression(
            penalty="l2",
            C=self.parameters.C,
            solver=self.parameters.solver,
            max_iter=100,
            n_jobs=1,
            class_weight="balanced",
        )


@dataclass
class PLSRegression(Algorithm):
    @dataclass
    class PLSParameters:
        n_components: int = field(metadata=schema(min=1))

    name: Literal["PLSRegression"]
    parameters: PLSParameters

    def estimator(self):
        return sklearn.cross_decomposition.PLSRegression(
            n_components=self.parameters.n_components
        )


@dataclass
class PLS(PLSRegression):
    # TODO: add FutureWarning
    name: Literal["PLS"]


@dataclass
class RandomForestClassifier(Algorithm):
    @dataclass
    class RandomForestParameters:
        max_depth: int = field(metadata=schema(min=1))
        n_estimators: int = field(metadata=schema(min=1))
        max_features: str

    name: Literal["RandomForestClassifier"]
    parameters: RandomForestParameters

    def estimator(self):
        return sklearn.ensemble.RandomForestClassifier(
            max_depth=self.parameters.max_depth,
            max_features=self.parameters.max_features,
            n_estimators=self.parameters.n_estimators,
            class_weight="balanced",
        )


@dataclass
class RandomForestRegressor(Algorithm):
    @dataclass
    class RandomForestParameters:
        max_depth: int = field(metadata=schema(min=1))
        n_estimators: int = field(metadata=schema(min=1))
        max_features: str

    name: Literal["RandomForestRegressor"]
    parameters: RandomForestParameters

    def estimator(self):
        return sklearn.ensemble.RandomForestRegressor(
            max_depth=self.parameters.max_depth,
            max_features=self.parameters.max_features,
            n_estimators=self.parameters.n_estimators,
        )


@dataclass
class RandomForest(RandomForestRegressor):
    # Deprecated
    pass


@dataclass
class Ridge(Algorithm):
    @dataclass
    class RidgeParameters:
        alpha: float = field(metadata=schema(min=0))

    name: Literal["Ridge"]
    parameters: RidgeParameters

    def estimator(self):
        return sklearn.linear_model.Ridge(alpha=self.parameters.alpha)


@dataclass
class SVC(Algorithm):
    @dataclass
    class SVCParameters:
        C: float = field(metadata=schema(min=1e-30, max=1e10))
        gamma: float = field(metadata=schema(min=1e-9, max=1e3))

    name: Literal["SVC"]
    parameters: SVCParameters

    def estimator(self):
        return sklearn.svm.SVC(
            C=self.parameters.C,
            gamma=self.parameters.gamma,
            class_weight="balanced",
        )


@dataclass
class SVR(Algorithm):
    @dataclass
    class SVRParameters:
        C: float = field(metadata=schema(min=1e-30, max=1e10))
        gamma: float = field(metadata=schema(min=1e-9, max=1e3))

    name: Literal["SVR"]
    parameters: SVRParameters

    def estimator(self):
        return sklearn.svm.SVR(C=self.parameters.C, gamma=self.parameters.gamma)


@dataclass
class XGBRegressor(Algorithm):
    @dataclass
    class XGBregressorParameters:
        max_depth: int = field(metadata=schema(min=1))
        n_estimators: int = field(metadata=schema(min=1))
        learning_rate: float = field(metadata=schema(min=0.0001))

    name: Literal["XGBRegressor"]
    parameters: XGBregressorParameters

    def estimator(self):
        return xgboost.XGBRegressor(
            max_depth=self.parameters.max_depth,
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            random_state=42,
            reg_lambda=1,
            objective="reg:squarederror",  # TODO: is this used, or the one in CV?
            subsample=1,
            booster="gbtree",
            verbosity=0,
            n_jobs=1,  # TODO: set to -1? Compete with CV.
            gamma=1,
        )


@dataclass
class XGBregressor(XGBRegressor):
    # Deprecated
    name: Literal["XGBregressor"]


AnyAlgorithm = Union[
    AdaBoostClassifier,
    Lasso,
    LogisticRegression,
    PLSRegression,
    PLS,  # Deprecated
    RandomForestRegressor,
    RandomForestClassifier,
    RandomForest,  # Deprecated
    Ridge,
    SVC,
    SVR,
    XGBRegressor,
    XGBregressor,  # Deprecated
]


@dataclass
class BuildConfig:
    """Build configuration.

    This is the configuration to train a model,
    i.e. optimize parameters of a model,
    given fixed hyperparameters.
    It roughly corresponds to Optuna Trial.
    """

    @dataclass
    class Metadata:
        cross_validation: Optional[int] = field(default=None, metadata=schema(min=1))
        shuffle: Optional[bool] = None
        best_trial: Optional[int] = field(default=None, metadata=schema(min=0))
        best_value: Optional[float] = None
        n_trials: Optional[int] = field(default=None, metadata=schema(min=0))
        visualization: Optional[Visualization] = None

    @dataclass
    class Settings:
        mode: Optional[ModelMode] = None
        scoring: Union[RegressionScore, ClassificationScore, str, None] = None
        direction: Optional[OptimizationDirection] = None
        n_trials: Optional[int] = field(default=None, metadata=schema(min=0))
        tracking_rest_endpoint: Optional[str] = field(
            default=None,
            metadata=schema(title="URL to track build results using internal format"),
        )

    data: Dataset
    metadata: Optional[Metadata]
    descriptor: AnyDescriptor
    settings: Optional[Settings]
    algorithm: AnyAlgorithm
    task: Literal["building"] = "building"
