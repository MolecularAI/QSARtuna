import sys
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

import optunaz
from optunaz import algorithms
from optunaz.algorithms import chem_prop
from optunaz.algorithms import chem_prop_hyperopt
from optunaz.algorithms import probabilistic_random_forest
from optunaz.config import (
    ModelMode,
    OptimizationDirection,
    Algorithm as GenericAlg,
    Visualization,
)
from optunaz.config.optconfig import (
    RegressionScore,
    ClassificationScore,
)
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
            estimator=None,
            random_state=42,
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            algorithm="SAMME",
        )


@dataclass
class Lasso(Algorithm):
    @dataclass
    class LassoParameters:
        alpha: float = field(default=1.0, metadata=schema(min=0))

    name: Literal["Lasso"]
    parameters: LassoParameters

    def estimator(self):
        return sklearn.linear_model.Lasso(alpha=self.parameters.alpha, random_state=42)


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
            random_state=42,
            C=self.parameters.C,
            solver=self.parameters.solver,
            max_iter=100,
            n_jobs=-1,
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
class RandomForestClassifier(Algorithm):
    @dataclass
    class RandomForestParameters:
        max_depth: int = field(metadata=schema(min=1))
        n_estimators: int = field(metadata=schema(min=1))
        max_features: str

    name: Literal["RandomForestClassifier"]
    parameters: RandomForestParameters

    def estimator(self):
        if self.parameters.max_features == "auto":
            max_features = 1.0
        else:
            max_features = self.parameters.max_features
        return sklearn.ensemble.RandomForestClassifier(
            max_depth=self.parameters.max_depth,
            max_features=max_features,
            n_estimators=self.parameters.n_estimators,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
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
        if self.parameters.max_features == "auto":
            max_features = 1.0
        else:
            max_features = self.parameters.max_features
        return sklearn.ensemble.RandomForestRegressor(
            max_depth=self.parameters.max_depth,
            max_features=max_features,
            n_estimators=self.parameters.n_estimators,
            random_state=42,
            n_jobs=-1,
        )


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
        C: float = field(default=1.0, metadata=schema(min=1e-30, max=1e10))
        gamma: float = field(default=1e-4, metadata=schema(min=1e-9, max=1e3))

    name: Literal["SVC"]
    parameters: SVCParameters

    def estimator(self):
        return sklearn.svm.SVC(
            C=self.parameters.C,
            gamma=self.parameters.gamma,
            class_weight="balanced",
            probability=True,
            random_state=42,
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
    class XGBRegressorParameters:
        max_depth: int = field(metadata=schema(min=1))
        n_estimators: int = field(metadata=schema(min=1))
        learning_rate: float = field(metadata=schema(min=0.0001))

    name: Literal["XGBRegressor"]
    parameters: XGBRegressorParameters

    def estimator(self):
        return xgboost.XGBRegressor(
            max_depth=self.parameters.max_depth,
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            random_state=42,
            reg_lambda=1,
            objective="reg:squarederror",
            subsample=1,
            booster="gbtree",
            verbosity=0,
            n_jobs=-1,
            gamma=1,
        )


@dataclass
class PRFClassifier(Algorithm):
    @dataclass
    class PRFClassifierParameters:
        max_depth: int = field(metadata=schema(min=1))
        n_estimators: int = field(metadata=schema(min=1))
        max_features: str
        use_py_gini: int = field(metadata=schema(default=1, min=0, max=1))
        use_py_leafs: int = field(metadata=schema(default=1, min=0, max=1))
        bootstrap: int = field(default=1, metadata=schema(min=0, max=1))
        new_syn_data_frac: float = field(default=0.0, metadata=schema(min=0))
        min_py_sum_leaf: int = field(default=1, metadata=schema(min=0))

    name: Literal["PRFClassifier"]
    parameters: PRFClassifierParameters

    def estimator(self):
        return optunaz.algorithms.probabilistic_random_forest.PRFClassifier(
            max_depth=self.parameters.max_depth,
            max_features=self.parameters.max_features,
            n_estimators=self.parameters.n_estimators,
            use_py_gini=self.parameters.use_py_gini,
            use_py_leafs=self.parameters.use_py_leafs,
            bootstrap=self.parameters.bootstrap,
            new_syn_data_frac=self.parameters.new_syn_data_frac,
            min_py_sum_leaf=self.parameters.min_py_sum_leaf,
        )


@dataclass
class ChemPropRegressor(Algorithm):
    @dataclass
    class ChemPropRegressorParameters:
        activation: str
        aggregation: str
        aggregation_norm: float = field(
            metadata=schema(min=1)
        )  # float: suggest_discrete_uniform fix
        batch_size: float = field(
            metadata=schema(min=5)
        )  # float: suggest_discrete_uniform fix
        depth: float = field(
            metadata=schema(min=2)
        )  # float: suggest_discrete_uniform fix
        dropout: float = field(metadata=schema(min=0))
        ensemble_size: int = field(metadata=schema(default=1, min=1, max=5))
        epochs: int = field(metadata=schema(default=30, min=4, max=400))
        features_generator: str
        ffn_hidden_size: float = field(
            metadata=schema(min=300)
        )  # float: suggest_discrete_uniform fix
        ffn_num_layers: float = field(
            metadata=schema(min=1)
        )  # float: suggest_discrete_uniform fix
        final_lr_ratio_exp: int = field(metadata=schema(min=-4))
        hidden_size: float = field(
            metadata=schema(min=300)
        )  # float: suggest_discrete_uniform fix
        init_lr_ratio_exp: int = field(metadata=schema(min=-4))
        max_lr_exp: int = field(metadata=schema(min=-6))
        warmup_epochs_ratio: float = field(default=0.1, metadata=schema(min=0, max=0.2))
        aux_weight_pc: int = 100

    name: Literal["ChemPropRegressor"]
    parameters: ChemPropRegressorParameters

    def estimator(self):
        return optunaz.algorithms.chem_prop.ChemPropRegressor(
            activation=self.parameters.activation,
            aggregation=self.parameters.aggregation,
            aggregation_norm=int(self.parameters.aggregation_norm),
            batch_size=int(self.parameters.batch_size),
            depth=int(self.parameters.depth),
            dropout=self.parameters.dropout,
            ensemble_size=self.parameters.ensemble_size,
            epochs=self.parameters.epochs,
            features_generator=self.parameters.features_generator,
            ffn_hidden_size=int(self.parameters.ffn_hidden_size),
            ffn_num_layers=int(self.parameters.ffn_num_layers),
            final_lr_ratio_exp=self.parameters.final_lr_ratio_exp,
            hidden_size=int(self.parameters.hidden_size),
            init_lr_ratio_exp=self.parameters.init_lr_ratio_exp,
            max_lr_exp=self.parameters.max_lr_exp,
            warmup_epochs_ratio=self.parameters.warmup_epochs_ratio,
            aux_weight_pc=self.parameters.aux_weight_pc,
        )


@dataclass
class ChemPropClassifier(Algorithm):
    @dataclass
    class ChemPropClassifierParameters:
        activation: str
        aggregation: str
        aggregation_norm: float = field(
            metadata=schema(min=1)
        )  # float: suggest_discrete_uniform fix
        batch_size: float = field(
            metadata=schema(min=5)
        )  # float: suggest_discrete_uniform fix
        depth: float = field(
            metadata=schema(min=2)
        )  # float: suggest_discrete_uniform fix
        dropout: float = field(metadata=schema(min=0))
        ensemble_size: int = field(metadata=schema(default=1, min=1, max=5))
        epochs: int = field(metadata=schema(default=30, min=4, max=400))
        features_generator: str
        ffn_hidden_size: float = field(
            metadata=schema(min=300)
        )  # float: suggest_discrete_uniform fix
        ffn_num_layers: float = field(
            metadata=schema(min=1)
        )  # float: suggest_discrete_uniform fix
        final_lr_ratio_exp: int = field(metadata=schema(min=-4))
        hidden_size: float = field(
            metadata=schema(min=300)
        )  # float: suggest_discrete_uniform fix
        init_lr_ratio_exp: int = field(metadata=schema(min=-4))
        max_lr_exp: int = field(metadata=schema(min=-6))
        warmup_epochs_ratio: float = field(default=0.1, metadata=schema(min=0, max=0.2))
        aux_weight_pc: int = 100

    name: Literal["ChemPropClassifier"]
    parameters: ChemPropClassifierParameters

    def estimator(self):
        return optunaz.algorithms.chem_prop.ChemPropClassifier(
            activation=self.parameters.activation,
            aggregation=self.parameters.aggregation,
            aggregation_norm=int(self.parameters.aggregation_norm),
            batch_size=int(self.parameters.batch_size),
            depth=int(self.parameters.depth),
            dropout=self.parameters.dropout,
            ensemble_size=self.parameters.ensemble_size,
            epochs=self.parameters.epochs,
            features_generator=self.parameters.features_generator,
            ffn_hidden_size=int(self.parameters.ffn_hidden_size),
            ffn_num_layers=int(self.parameters.ffn_num_layers),
            final_lr_ratio_exp=self.parameters.final_lr_ratio_exp,
            hidden_size=int(self.parameters.hidden_size),
            init_lr_ratio_exp=self.parameters.init_lr_ratio_exp,
            max_lr_exp=self.parameters.max_lr_exp,
            warmup_epochs_ratio=self.parameters.warmup_epochs_ratio,
            aux_weight_pc=self.parameters.aux_weight_pc,
        )


@dataclass
class ChemPropHyperoptClassifier(Algorithm):
    @dataclass
    class ChemPropHyperoptClassifierParameters:
        ensemble_size: int = field(metadata=schema(default=1, min=1, max=5))
        epochs: int = field(metadata=schema(default=30, min=4, max=400))
        features_generator: str
        num_iters: int = field(metadata=schema(default=30, min=1, max=50))
        search_parameter_level: str
        aux_weight_pc: int = 100

    name: Literal["ChemPropHyperoptClassifier"]
    parameters: ChemPropHyperoptClassifierParameters

    def estimator(self):
        return optunaz.algorithms.chem_prop_hyperopt.ChemPropHyperoptClassifier(
            ensemble_size=self.parameters.ensemble_size,
            epochs=self.parameters.epochs,
            features_generator=self.parameters.features_generator,
            num_iters=self.parameters.num_iters,
            search_parameter_level=self.parameters.search_parameter_level,
            aux_weight_pc=self.parameters.aux_weight_pc,
        )


@dataclass
class ChemPropHyperoptRegressor(Algorithm):
    @dataclass
    class ChemPropHyperoptRegressorParameters:
        ensemble_size: int = field(metadata=schema(default=1, min=1, max=5))
        epochs: int = field(metadata=schema(default=30, min=4, max=400))
        features_generator: str
        num_iters: int = field(metadata=schema(default=30, min=1, max=50))
        search_parameter_level: str
        aux_weight_pc: int = 100

    name: Literal["ChemPropHyperoptRegressor"]
    parameters: ChemPropHyperoptRegressorParameters

    def estimator(self):
        return optunaz.algorithms.chem_prop_hyperopt.ChemPropHyperoptRegressor(
            ensemble_size=self.parameters.ensemble_size,
            epochs=self.parameters.epochs,
            features_generator=self.parameters.features_generator,
            num_iters=self.parameters.num_iters,
            search_parameter_level=self.parameters.search_parameter_level,
            aux_weight_pc=self.parameters.aux_weight_pc,
        )


AnyUncalibratedClassifier = Union[
    AdaBoostClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    ChemPropClassifier,
    ChemPropRegressor,
    ChemPropHyperoptClassifier,
    ChemPropHyperoptRegressor,
]


@dataclass
class CalibratedClassifierCVWithVA(Algorithm):
    @dataclass
    class CalibratedClassifierCVParameters:
        n_folds: int = field(metadata=schema(default=2, min=2))
        ensemble: str
        method: str
        estimator: AnyUncalibratedClassifier

    name: Literal["CalibratedClassifierCVWithVA"]
    parameters: CalibratedClassifierCVParameters

    def __post_init__(self):
        # pop sys calibration cache for monkey patch
        sys.modules.pop("sklearn.calibration", None)
        import optunaz.algorithms.calibrated_cv

    def estimator(self):
        # pop sys calibration cache for monkey patch
        sys.modules.pop("sklearn.calibration", None)
        import optunaz.algorithms.calibrated_cv

        return optunaz.algorithms.calibrated_cv.CalibratedClassifierCVWithVA(
            self.parameters.estimator.estimator(),
            n_folds=self.parameters.n_folds,
            ensemble=self.parameters.ensemble == "True",
            method=self.parameters.method,
            n_jobs=-1,
        )


AnyRegression = Union[
    Lasso,
    PLSRegression,
    RandomForestRegressor,
    Ridge,
    SVR,
    XGBRegressor,
    PRFClassifier,
    ChemPropRegressor,
    ChemPropHyperoptRegressor,
]

MapieCompatible = Union[
    Lasso,
    PLSRegression,
    RandomForestRegressor,
    Ridge,
    SVR,
    XGBRegressor,
    PRFClassifier,
]


@dataclass
class Mapie(Algorithm):
    @dataclass
    class MapieParameters:
        alpha: float = field(metadata=schema(default=0.05, min=0.01, max=0.99))
        estimator: MapieCompatible

    name: Literal["Mapie"]
    parameters: MapieParameters

    def estimator(self):
        from optunaz.algorithms.mapie_uncertainty import MapieWithUncertainty

        return MapieWithUncertainty(
            alpha=self.parameters.alpha,
            estimator=self.parameters.estimator.estimator(),
            n_jobs=-1,
        )


AnyAlgorithm = Union[
    AnyUncalibratedClassifier,
    AnyRegression,
    CalibratedClassifierCVWithVA,
    Mapie,
]

AnyChemPropAlgorithm = [
    ChemPropClassifier,
    ChemPropRegressor,
    ChemPropHyperoptClassifier,
    ChemPropHyperoptRegressor,
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
        name: Optional[str] = None
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
