import abc
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import dill
import sklearn
import sklearn.cross_decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.svm
import xgboost
from apischema import schema
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier

import optunaz
from optunaz import algorithms
from optunaz.algorithms import (
    calibrated_cv,
    chem_prop,
    fast_prop,
    probabilistic_random_forest,
    tabpfn,
    custom_catboost
)
from optunaz.config import Algorithm as GenericAlg
from optunaz.config import ModelMode, OptimizationDirection, Visualization
from optunaz.config.optconfig import ClassificationScore, RegressionScore
from optunaz.datareader import Dataset
from optunaz.descriptors import AnyDescriptor
from optunaz.utils.files_paths import copy_path_for_scaled_descriptor


class Algorithm(GenericAlg):
    @abc.abstractmethod
    def estimator(self) -> BaseEstimator:
        pass


@dataclass
class AdaBoostClassifier(Algorithm):
    @dataclass
    class AdaBoostClassifierParameters:
        n_estimators: int = field(default=1, metadata=schema(min=1))
        learning_rate: float = field(default=1.0, metadata=schema(min=0.0001))
        max_depth: int = field(default=1, metadata=schema(min=1))

    name: Literal["AdaBoostClassifier"] = "AdaBoostClassifier"
    parameters: AdaBoostClassifierParameters = field(
        default_factory=AdaBoostClassifierParameters
    )

    def estimator(self):
        return sklearn.ensemble.AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=self.parameters.max_depth, random_state=42
            ),
            random_state=42,
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
        )


@dataclass
class CatBoostClassifier(Algorithm):
    @dataclass
    class CatBoostClassifierParameters:
        n_estimators: int = field(default=500, metadata=schema(min=1))
        learning_rate: float = field(default=0.03, metadata=schema(min=0.0001))
        depth: int = field(default=6, metadata=schema(min=1))
        l2_leaf_reg: float = field(default=3.0, metadata=schema(min=0.0001))
        random_strength: float = field(default=1.0, metadata=schema(min=0.0001))

    name: Literal["CatBoostClassifier"] = "CatBoostClassifier"
    parameters: CatBoostClassifierParameters = field(
        default_factory=CatBoostClassifierParameters
    )

    def estimator(self):
        return custom_catboost.CustomCatBoostClassifier(
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            depth=self.parameters.depth,
            l2_leaf_reg=self.parameters.l2_leaf_reg,
            random_strength=self.parameters.random_strength,
        )


@dataclass
class CatBoostRegressor(Algorithm):
    @dataclass
    class CatBoostRegressorParameters:
        n_estimators: int = field(default=500, metadata=schema(min=1))
        learning_rate: float = field(default=0.03, metadata=schema(min=0.0001))
        depth: int = field(default=6, metadata=schema(min=1))
        l2_leaf_reg: float = field(default=3.0, metadata=schema(min=0.0001))
        random_strength: float = field(default=1.0, metadata=schema(min=0.0001))

    name: Literal["CatBoostRegressor"] = "CatBoostRegressor"
    parameters: CatBoostRegressorParameters = field(
        default_factory=CatBoostRegressorParameters
    )

    def estimator(self):
        return custom_catboost.CustomCatBoostRegressor(
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            depth=self.parameters.depth,
            l2_leaf_reg=self.parameters.l2_leaf_reg,
            random_strength=self.parameters.random_strength,
        )


@dataclass
class Lasso(Algorithm):
    @dataclass
    class LassoParameters:
        alpha: float = field(default=1.0, metadata=schema(min=0))
        max_iter: int = field(default=1000, metadata=schema(min=1000))
        tol: float = field(default=1e-4, metadata=schema(min=1e-4))

    name: Literal["Lasso"] = "Lasso"
    parameters: LassoParameters = field(default_factory=LassoParameters)

    def estimator(self):
        return sklearn.linear_model.Lasso(
            alpha=self.parameters.alpha,
            random_state=42,
            max_iter=self.parameters.max_iter,
            tol=self.parameters.tol,
        )


@dataclass
class KNeighborsClassifier(Algorithm):
    @dataclass
    class KNeighborsClassifierParameters:
        metric: str = field(default="minkowski")
        weights: str = field(default="uniform")
        n_neighbors: int = field(default=5, metadata=schema(min=1))

    name: Literal["KNeighborsClassifier"] = "KNeighborsClassifier"
    parameters: KNeighborsClassifierParameters = field(
        default_factory=KNeighborsClassifierParameters
    )

    def estimator(self):
        return sklearn.neighbors.KNeighborsClassifier(
            metric=self.parameters.metric,
            n_jobs=-1,
            n_neighbors=self.parameters.n_neighbors,
            weights=self.parameters.weights,
        )


@dataclass
class KNeighborsRegressor(Algorithm):
    @dataclass
    class KNeighborsRegressorParameters:
        metric: str = field(default="minkowski")
        weights: str = field(default="uniform")
        n_neighbors: int = field(default=5, metadata=schema(min=1))

    name: Literal["KNeighborsRegressor"] = "KNeighborsRegressor"
    parameters: KNeighborsRegressorParameters = field(
        default_factory=KNeighborsRegressorParameters
    )

    def estimator(self):
        return sklearn.neighbors.KNeighborsRegressor(
            metric=self.parameters.metric,
            n_jobs=-1,
            n_neighbors=self.parameters.n_neighbors,
            weights=self.parameters.weights,
        )


@dataclass
class LogisticRegression(Algorithm):
    @dataclass
    class LogisticRegressionParameters:
        solver: str = field(default="lbfgs")
        penalty: str = field(default="l2")
        C: float = field(default=1.0, metadata=schema(min=0.001, max=1000))

    name: Literal["LogisticRegression"] = "LogisticRegression"
    parameters: LogisticRegressionParameters = field(
        default_factory=LogisticRegressionParameters
    )

    def estimator(self):
        return sklearn.linear_model.LogisticRegression(
            penalty=self.parameters.penalty,
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
        n_components: int = field(default=2, metadata=schema(min=1))

    name: Literal["PLSRegression"] = "PLSRegression"
    parameters: PLSParameters = field(default_factory=PLSParameters)

    def estimator(self):
        return sklearn.cross_decomposition.PLSRegression(
            n_components=self.parameters.n_components
        )


@dataclass
class RandomForestClassifier(Algorithm):
    @dataclass
    class RandomForestParameters:
        max_features: str = field(default="auto")
        max_depth: Optional[int] = field(default=None)
        n_estimators: int = field(default=100, metadata=schema(min=1))

    name: Literal["RandomForestClassifier"] = "RandomForestClassifier"
    parameters: RandomForestParameters = field(default_factory=RandomForestParameters)

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
        max_depth: Optional[int] = field(default=None)
        n_estimators: int = field(default=100, metadata=schema(min=1))
        max_features: str = field(default="auto")

    name: Literal["RandomForestRegressor"] = "RandomForestRegressor"
    parameters: RandomForestParameters = field(default_factory=RandomForestParameters)

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
        alpha: float = field(default=1, metadata=schema(min=0))

    name: Literal["Ridge"] = "Ridge"
    parameters: RidgeParameters = field(default_factory=RidgeParameters)

    def estimator(self):
        return sklearn.linear_model.Ridge(alpha=self.parameters.alpha)


@dataclass
class SVC(Algorithm):
    @dataclass
    class SVCParameters:
        C: float = field(default=1.0, metadata=schema(min=1e-30, max=1e10))
        gamma: float = field(default=1e-4, metadata=schema(min=1e-9, max=1e3))

    name: Literal["SVC"] = "SVC"
    parameters: SVCParameters = field(default_factory=SVCParameters)

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
        C: float = field(default=1.0, metadata=schema(min=1e-30, max=1e10))
        gamma: float = field(default=1e-4, metadata=schema(min=1e-9, max=1e3))

    name: Literal["SVR"] = "SVR"
    parameters: SVRParameters = field(default_factory=SVRParameters)

    def estimator(self):
        return sklearn.svm.SVR(C=self.parameters.C, gamma=self.parameters.gamma)


@dataclass
class XGBClassifier(Algorithm):
    @dataclass
    class XGBClassifierParameters:
        max_depth: int = field(default=2, metadata=schema(min=1))
        n_estimators: int = field(default=100, metadata=schema(min=1))
        learning_rate: float = field(default=1.0, metadata=schema(min=0.0001))
        subsample: float = field(default=1.0, metadata=schema(min=0.01))
        gamma: float = field(default=1e-3, metadata=schema(min=0.0))
        colsample_bytree: float = field(default=1.0, metadata=schema(min=0.0001))

    name: Literal["XGBClassifier"] = "XGBClassifier"
    parameters: XGBClassifierParameters = field(default_factory=XGBClassifierParameters)

    def estimator(self):
        return xgboost.XGBClassifier(
            max_depth=self.parameters.max_depth,
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            random_state=42,
            reg_lambda=1,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=self.parameters.subsample,
            booster="gbtree",
            verbosity=0,
            n_jobs=-1,
            gamma=self.parameters.gamma,
            colsample_bytree=self.parameters.colsample_bytree,
        )


@dataclass
class XGBRegressor(Algorithm):
    @dataclass
    class XGBRegressorParameters:
        max_depth: int = field(default=2, metadata=schema(min=1))
        n_estimators: int = field(default=100, metadata=schema(min=1))
        learning_rate: float = field(default=1.0, metadata=schema(min=0.0001))
        subsample: float = field(default=1.0, metadata=schema(min=0.01))
        gamma: float = field(default=1e-3, metadata=schema(min=0.0))
        colsample_bytree: float = field(default=1.0, metadata=schema(min=0.0001))

    name: Literal["XGBRegressor"] = "XGBRegressor"
    parameters: XGBRegressorParameters = field(default_factory=XGBRegressorParameters)

    def estimator(self):
        return xgboost.XGBRegressor(
            max_depth=self.parameters.max_depth,
            n_estimators=self.parameters.n_estimators,
            learning_rate=self.parameters.learning_rate,
            random_state=42,
            reg_lambda=1,
            objective="reg:squarederror",
            subsample=self.parameters.subsample,
            booster="gbtree",
            verbosity=0,
            n_jobs=-1,
            gamma=self.parameters.gamma,
            colsample_bytree=self.parameters.colsample_bytree,
        )


@dataclass
class PRFClassifier(Algorithm):
    @dataclass
    class PRFClassifierParameters:
        max_depth: int = field(default=2, metadata=schema(min=1))
        n_estimators: int = field(default=100, metadata=schema(min=1))
        use_py_gini: int = field(default=1, metadata=schema(default=1, min=0, max=1))
        use_py_leafs: int = field(default=1, metadata=schema(default=1, min=0, max=1))
        bootstrap: int = field(default=1, metadata=schema(min=0, max=1))
        new_syn_data_frac: float = field(default=0.0, metadata=schema(min=0))
        min_py_sum_leaf: int = field(default=1, metadata=schema(min=0))
        max_features: str = field(default="auto")

    name: Literal["PRFClassifier"] = "PRFClassifier"
    parameters: PRFClassifierParameters = field(default_factory=PRFClassifierParameters)

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
class TabPFNClassifier(Algorithm):
    @dataclass
    class TabPFNClassifierParameters:
        max_time: int = field(default=150, metadata=schema(min=1))
        random_state: int = field(default=42, metadata=schema(min=0))
        max_feats: int = field(default=500, metadata=schema(min=1))
        feature_selection: str = field(default="k_best")
        eval_metric: str = field(default="accuracy")

    name: Literal["TabPFNClassifier"] = "TabPFNClassifier"
    parameters: TabPFNClassifierParameters = field(
        default_factory=TabPFNClassifierParameters
    )

    def estimator(self):
        return optunaz.algorithms.tabpfn.TabPFNClassifier(
            max_time=self.parameters.max_time,
            random_state=self.parameters.random_state,
            max_feats=self.parameters.max_feats,
            feature_selection=self.parameters.feature_selection,
            eval_metric=self.parameters.eval_metric,
        )


@dataclass
class TabPFNRegressor(Algorithm):
    @dataclass
    class TabPFNRegressorParameters:
        max_time: int = field(default=150, metadata=schema(min=1))
        random_state: int = field(default=42, metadata=schema(min=0))
        max_feats: int = field(default=500, metadata=schema(min=1))
        feature_selection: str = field(default="k_best")
        eval_metric: str = field(default="root_mean_squared_error")

    name: Literal["TabPFNRegressor"] = "TabPFNRegressor"
    parameters: TabPFNRegressorParameters = field(
        default_factory=TabPFNRegressorParameters
    )

    def estimator(self):
        return optunaz.algorithms.tabpfn.TabPFNRegressor(
            max_time=self.parameters.max_time,
            random_state=self.parameters.random_state,
            max_feats=self.parameters.max_feats,
            feature_selection=self.parameters.feature_selection,
            eval_metric=self.parameters.eval_metric,
        )


@dataclass
class FastPropClassifier(Algorithm):
    @dataclass
    class FastPropClassifierParameters:
        batch_size: int = field(default=64, metadata=schema(min=16))
        number_epochs: int = field(default=100, metadata=schema(min=1))
        number_repeats: int = field(default=1, metadata=schema(min=1))
        patience: int = field(default=10, metadata=schema(min=0))
        hidden_size: int = field(default=1800, metadata=schema(min=100))
        fnn_layers: int = field(default=2, metadata=schema(min=1))
        learning_rate: float = field(default=0.0001, metadata=schema(min=1e-05))
        random_seed: int = field(default=42, metadata=schema(min=1))
        train_size: float = field(default=0.9, metadata=schema(min=1e-05))
        val_size: float = field(default=0.15, metadata=schema(min=1e-05))
        test_size: float = field(default=0.05, metadata=schema(min=1e-05))

    name: Literal["FastPropClassifier"] = "FastPropClassifier"
    parameters: FastPropClassifierParameters = field(
        default_factory=FastPropClassifierParameters
    )

    def estimator(self):
        return optunaz.algorithms.fast_prop.FastPropClassifier(
            batch_size=self.parameters.batch_size,
            number_epochs=self.parameters.number_epochs,
            number_repeats=self.parameters.number_repeats,
            patience=self.parameters.patience,
            hidden_size=self.parameters.hidden_size,
            fnn_layers=self.parameters.fnn_layers,
            learning_rate=self.parameters.learning_rate,
            val_size=self.parameters.val_size,
            test_size=self.parameters.test_size,
            train_size=self.parameters.train_size,
        )


@dataclass
class FastPropRegressor(Algorithm):
    @dataclass
    class FastPropRegressorParameters:
        batch_size: int = field(default=64, metadata=schema(min=16))
        number_epochs: int = field(default=100, metadata=schema(min=1))
        number_repeats: int = field(default=1, metadata=schema(min=1))
        patience: int = field(default=10, metadata=schema(min=0))
        hidden_size: int = field(default=1800, metadata=schema(min=100))
        fnn_layers: int = field(default=2, metadata=schema(min=1))
        learning_rate: float = field(default=0.0001, metadata=schema(min=1e-05))
        random_seed: int = field(default=42, metadata=schema(min=1))
        train_size: float = field(default=0.9, metadata=schema(min=1e-05))
        val_size: float = field(default=0.15, metadata=schema(min=1e-05))
        test_size: float = field(default=0.05, metadata=schema(min=1e-05))

    name: Literal["FastPropRegressor"] = "FastPropRegressor"
    parameters: FastPropRegressorParameters = field(
        default_factory=FastPropRegressorParameters
    )

    def estimator(self):
        return optunaz.algorithms.fast_prop.FastPropRegressor(
            batch_size=self.parameters.batch_size,
            number_epochs=self.parameters.number_epochs,
            number_repeats=self.parameters.number_repeats,
            patience=self.parameters.patience,
            hidden_size=self.parameters.hidden_size,
            fnn_layers=self.parameters.fnn_layers,
            learning_rate=self.parameters.learning_rate,
            val_size=self.parameters.val_size,
            test_size=self.parameters.test_size,
            train_size=self.parameters.train_size,
        )


@dataclass
class ChemPropRegressor(Algorithm):
    @dataclass
    class ChemPropRegressorParameters:
        aggregation_norm: int = field(default=100, metadata=schema(min=1))
        batch_size: int = field(default=64, metadata=schema(min=16))
        depth: int = field(default=3, metadata=schema(min=2))
        dropout: float = field(default=0.0, metadata=schema(min=0.0))
        ensemble_size: int = field(default=1, metadata=schema(min=1))
        epochs: int = field(default=100, metadata=schema(min=1))
        patience: int = field(default=10, metadata=schema(min=0))
        ffn_hidden_dim: int = field(default=300, metadata=schema(min=300))
        ffn_num_layers: int = field(default=1, metadata=schema(min=1))
        final_lr_ratio: float = field(default=1e-04, metadata=schema(min=1e-06))
        message_hidden_dim: int = field(default=300, metadata=schema(min=300))
        init_lr_ratio: float = field(default=1e-04, metadata=schema(min=1e-06))
        max_lr: float = field(default=1e-03, metadata=schema(min=1e-03))
        warmup_epochs_ratio: float = field(default=0.01, metadata=schema(min=0.0))
        y_aux_weight_pc: int = field(default=100, metadata=schema(min=0, max=100))
        activation: str = field(default="RELU")
        aggregation: str = field(default="norm")
        loss_function: str = field(default="mse")
        batch_norm: bool | int = field(default=False)
        message_bias: bool | int = field(default=False)
        undirected: bool | int = field(default=False)
        molecule_featurizers: list | None = field(default=None)

    name: Literal["ChemPropRegressor"] = "ChemPropRegressor"
    parameters: ChemPropRegressorParameters = field(
        default_factory=ChemPropRegressorParameters
    )

    def estimator(self):
        return optunaz.algorithms.chem_prop.ChemPropRegressor(
            activation=self.parameters.activation,
            aggregation=self.parameters.aggregation,
            aggregation_norm=self.parameters.aggregation_norm,
            batch_size=self.parameters.batch_size,
            batch_norm=bool(self.parameters.batch_norm),
            depth=self.parameters.depth,
            dropout=self.parameters.dropout,
            ensemble_size=self.parameters.ensemble_size,
            epochs=self.parameters.epochs,
            molecule_featurizers=self.parameters.molecule_featurizers,
            ffn_hidden_dim=self.parameters.ffn_hidden_dim,
            ffn_num_layers=self.parameters.ffn_num_layers,
            final_lr_ratio=self.parameters.final_lr_ratio,
            message_hidden_dim=self.parameters.message_hidden_dim,
            message_bias=bool(self.parameters.message_bias),
            loss_function=self.parameters.loss_function,
            init_lr_ratio=self.parameters.init_lr_ratio,
            max_lr=self.parameters.max_lr,
            warmup_epochs_ratio=self.parameters.warmup_epochs_ratio,
            y_aux_weight_pc=self.parameters.y_aux_weight_pc,
            undirected=bool(self.parameters.undirected),
        )


@dataclass
class ChemPropClassifier(Algorithm):
    @dataclass
    class ChemPropClassifierParameters:
        aggregation_norm: int = field(default=100, metadata=schema(min=1))
        batch_size: int = field(default=64, metadata=schema(min=16))
        depth: int = field(default=3, metadata=schema(min=2))
        dropout: float = field(default=0.0, metadata=schema(min=0.0))
        ensemble_size: int = field(default=1, metadata=schema(min=1))
        epochs: int = field(default=100, metadata=schema(min=1))
        patience: int = field(default=10, metadata=schema(min=0))
        ffn_hidden_dim: int = field(default=300, metadata=schema(min=300))
        ffn_num_layers: int = field(default=1, metadata=schema(min=1))
        final_lr_ratio: float = field(default=1e-04, metadata=schema(min=1e-06))
        message_hidden_dim: int = field(default=300, metadata=schema(min=300))
        init_lr_ratio: float = field(default=1e-04, metadata=schema(min=1e-06))
        max_lr: float = field(default=1e-03, metadata=schema(min=1e-03))
        warmup_epochs_ratio: float = field(default=0.01, metadata=schema(min=0.0))
        y_aux_weight_pc: int = field(default=100, metadata=schema(min=0, max=100))
        activation: str = field(default="RELU")
        aggregation: str = field(default="norm")
        loss_function: str = field(default="bce")
        batch_norm: bool | int = field(default=False)
        message_bias: bool | int = field(default=False)
        undirected: bool | int = field(default=False)
        molecule_featurizers: list | None = field(default=None)

    name: Literal["ChemPropClassifier"] = "ChemPropClassifier"
    parameters: ChemPropClassifierParameters = field(
        default_factory=ChemPropClassifierParameters
    )

    def estimator(self):
        return optunaz.algorithms.chem_prop.ChemPropClassifier(
            activation=self.parameters.activation,
            aggregation=self.parameters.aggregation,
            aggregation_norm=self.parameters.aggregation_norm,
            batch_size=self.parameters.batch_size,
            batch_norm=bool(self.parameters.batch_norm),
            depth=self.parameters.depth,
            dropout=self.parameters.dropout,
            ensemble_size=self.parameters.ensemble_size,
            epochs=self.parameters.epochs,
            molecule_featurizers=self.parameters.molecule_featurizers,
            ffn_hidden_dim=self.parameters.ffn_hidden_dim,
            ffn_num_layers=self.parameters.ffn_num_layers,
            final_lr_ratio=self.parameters.final_lr_ratio,
            message_hidden_dim=self.parameters.message_hidden_dim,
            message_bias=bool(self.parameters.message_bias),
            loss_function=self.parameters.loss_function,
            init_lr_ratio=self.parameters.init_lr_ratio,
            max_lr=self.parameters.max_lr,
            warmup_epochs_ratio=self.parameters.warmup_epochs_ratio,
            y_aux_weight_pc=self.parameters.y_aux_weight_pc,
            undirected=bool(self.parameters.undirected),
        )


@dataclass
class ChemPropRegressorPretrained(Algorithm):
    @dataclass
    class ChemPropRegressorPretrainedParameters:
        epochs: int = field(default=100, metadata=schema(min=0))
        patience: int = field(default=10, metadata=schema(min=0))
        frzn: str | None = field(default=None)
        pretrained_model: str = field(default=None)

    name: Literal["ChemPropRegressorPretrained"] = "ChemPropRegressorPretrained"
    parameters: ChemPropRegressorPretrainedParameters = field(
        default_factory=ChemPropRegressorPretrainedParameters
    )

    def estimator(self):
        return optunaz.algorithms.chem_prop.ChemPropRegressorPretrained(
            epochs=self.parameters.epochs,
            patience=self.parameters.patience,
            frzn=self.parameters.frzn,
            pretrained_model=self.parameters.pretrained_model,
        )


@dataclass
class CustomClassificationModel(Algorithm):
    @dataclass
    class CustomClassificationModelParameters:
        refit_model: int = field(default=0, metadata=schema(min=0, max=1))
        model_file: str = field(default=None)

    class CustomClassificationEstimator(ClassifierMixin, BaseEstimator):
        def __init__(self, model_file: str, refit_model: int):
            self.model_file = model_file
            self.refit_model = refit_model
            self.model = None

        def _load_model(self):
            """Load the pre-existing model from the file."""
            if self.model is None:
                with open(self.model_file, "rb") as fid:
                    model = dill.load(fid)

                # Import the QptunaModel class (not provided here but make sure to import it)
                from optunaz.model_writer import QptunaModel

                if isinstance(model, QptunaModel):
                    model = model.predictor
                if isinstance(
                    model,
                    CustomClassificationModel.CustomClassificationEstimator,
                ):
                    model = model.model

                if not hasattr(model, "predict_proba"):
                    raise ValueError(
                        "An estimator with a 'predict_proba' method must be supplied."
                    )

                self.model = model

        def fit(self, X, y):
            """Fits the model data, if necessary."""
            self._load_model()
            if self.refit_model:
                self.model.fit(X, y)
                self.X_ = X
                self.y_ = y
            else:
                self.X_ = None
                self.y_ = None
            self.__sklearn_is_fitted__ = True
            return self

        def predict(self, X):
            """Predicts using the pre-existing model."""
            self._load_model()
            return self.model.predict(X)

        def predict_proba(self, X):
            """Predicts proba using the pre-existing model."""
            self._load_model()
            return self.model.predict_proba(X)

        def __getattr__(self, attr):
            """Delegate attribute access to the pre-existing model."""
            self._load_model()
            return getattr(self.model, attr)

        def get_params(self, deep=True) -> dict:
            """Returns parameters for this estimator."""
            return {
                "model_file": self.model_file,
                "refit_model": self.refit_model,
            }

        def set_params(self, **params) -> "CustomClassificationEstimator":
            """Sets the parameters of this estimator."""
            for param, value in params.items():
                setattr(self, param, value)
            self.model = None  # Reset model to ensure it reloads
            return self

        def __sklearn_clone__(self):
            """Custom method to clone the estimator."""
            clone_estimator = self.__class__(
                model_file=self.model_file,
                refit_model=self.refit_model,
            )
            return clone_estimator

    name: Literal["CustomClassificationModel"] = "CustomClassificationModel"
    parameters: CustomClassificationModelParameters = field(
        default_factory=CustomClassificationModelParameters
    )

    def estimator(self) -> CustomClassificationEstimator:
        """Creates and returns a custom regression estimator."""
        return self.CustomClassificationEstimator(
            model_file=self.parameters.model_file,
            refit_model=self.parameters.refit_model,
        )


@dataclass
class CustomRegressionModel(Algorithm):
    @dataclass
    class CustomRegressionModelParameters:
        refit_model: int = field(default=0, metadata=schema(min=0, max=1))
        model_file: str = field(default=None)

    class CustomRegressionEstimator(RegressorMixin, BaseEstimator):
        def __init__(self, model_file: str, refit_model: int):
            self.model_file = model_file
            self.refit_model = refit_model
            self.model = None

        def _load_model(self):
            """Load the pre-existing model from the file."""
            if self.model is None:
                with open(self.model_file, "rb") as fid:
                    model = dill.load(fid)

                # Import the QptunaModel class (not provided here but make sure to import it)
                from optunaz.model_writer import QptunaModel

                if isinstance(model, QptunaModel):
                    model = model.predictor
                if isinstance(model, CustomRegressionModel.CustomRegressionEstimator):
                    model = model.model

                if not hasattr(model, "predict"):
                    raise ValueError(
                        "An estimator with a 'predict' method must be supplied."
                    )

                self.model = model

        def fit(self, X, y):
            """Fits the model data, if necessary."""
            self._load_model()
            if self.refit_model:
                self.model.fit(X, y)
                self.X_ = X
                self.y_ = y
            else:
                self.X_ = None
                self.y_ = None
            self.__sklearn_is_fitted__ = True
            return self

        def predict(self, X):
            """Predicts using the pre-existing model."""
            self._load_model()
            return self.model.predict(X)

        def __getattr__(self, attr):
            """Delegate attribute access to the pre-existing model."""
            self._load_model()
            return getattr(self.model, attr)

        def get_params(self, deep=True) -> dict:
            """Returns parameters for this estimator."""
            return {
                "model_file": self.model_file,
                "refit_model": self.refit_model,
            }

        def set_params(self, **params) -> "CustomRegressionEstimator":
            """Sets the parameters of this estimator."""
            for param, value in params.items():
                setattr(self, param, value)
            self.model = None  # Reset model to ensure it reloads
            return self

        def __sklearn_clone__(self):
            """Custom method to clone the estimator."""
            clone_estimator = self.__class__(
                model_file=self.model_file,
                refit_model=self.refit_model,
            )
            return clone_estimator

    name: Literal["CustomRegressionModel"] = "CustomRegressionModel"
    parameters: CustomRegressionModelParameters = field(
        default_factory=CustomRegressionModelParameters
    )

    def estimator(self) -> CustomRegressionEstimator:
        """Creates and returns a custom regression estimator."""
        return self.CustomRegressionEstimator(
            model_file=self.parameters.model_file,
            refit_model=self.parameters.refit_model,
        )


AnyUncalibratedClassifier = Union[
    AdaBoostClassifier,
    CatBoostClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    RandomForestClassifier,
    XGBClassifier,
    SVC,
    TabPFNClassifier,
    FastPropClassifier,
    ChemPropClassifier,
    CustomClassificationModel,
]

AvoidNestedParallelism = Union[
    ChemPropClassifier,
    ChemPropRegressor,
    ChemPropRegressorPretrained,
    TabPFNClassifier,
    TabPFNRegressor,
    FastPropClassifier,
    FastPropRegressor,
    CustomClassificationModel,
    CustomRegressionModel,
]


@dataclass
class CalibratedClassifierCVWithVA(Algorithm):
    @dataclass
    class CalibratedClassifierCVParameters:
        n_folds: int = field(default=5, metadata=schema(min=2))
        estimator: AnyUncalibratedClassifier = field(default=None)
        ensemble: str = field(default="True")
        method: str = field(default="sigmoid")

    name: Literal["CalibratedClassifierCVWithVA"] = "CalibratedClassifierCVWithVA"
    parameters: CalibratedClassifierCVParameters = field(
        default_factory=CalibratedClassifierCVParameters
    )

    def estimator(self):
        return optunaz.algorithms.calibrated_cv.CalibratedClassifierCVWithVA(
            estimator=self.parameters.estimator.estimator(),
            n_folds=self.parameters.n_folds,
            ensemble=self.parameters.ensemble == "True",
            method=self.parameters.method,
            n_jobs=get_n_jobs(self.parameters.estimator),
        )


AnyRegression = Union[
    CatBoostRegressor,
    Lasso,
    PLSRegression,
    RandomForestRegressor,
    Ridge,
    KNeighborsRegressor,
    SVR,
    XGBRegressor,
    PRFClassifier,
    TabPFNRegressor,
    FastPropRegressor,
    ChemPropRegressor,
    ChemPropRegressorPretrained,
    CustomRegressionModel,
]

MapieRegressorCompatible = Union[
    CatBoostRegressor,
    Lasso,
    PLSRegression,
    RandomForestRegressor,
    KNeighborsRegressor,
    Ridge,
    SVR,
    XGBRegressor,
    PRFClassifier,
    CustomRegressionModel,
    TabPFNRegressor,
    FastPropRegressor,
]

MapieClassifierCompatible = Union[
    AnyUncalibratedClassifier,
    CalibratedClassifierCVWithVA,
]


@dataclass
class MapieRegressor(Algorithm):
    @dataclass
    class MapieRegressorParameters:
        mapie_alpha: float = field(default=0.05, metadata=schema(min=0.01))
        test_size: float = field(default=0.1, metadata=schema(min=0.01))
        estimator: MapieRegressorCompatible = field(default=None)
        n_folds: int = field(default=5, metadata=schema(min=2))
        random_state: int = field(default=42, metadata=schema(min=0))

    name: Literal["MapieRegressor"] = "MapieRegressor"
    parameters: MapieRegressorParameters = field(
        default_factory=MapieRegressorParameters
    )

    def estimator(self):
        from optunaz.algorithms.mapie_uncertainty import MapieRegressorWithUncertainty

        return MapieRegressorWithUncertainty(
            mapie_alpha=self.parameters.mapie_alpha,
            test_size=self.parameters.test_size,
            estimator=self.parameters.estimator.estimator(),
            n_jobs=get_n_jobs(self.parameters.estimator),
            cv=self.parameters.n_folds,
            random_state=self.parameters.random_state,
        )


@dataclass
class MapieClassifier(Algorithm):
    @dataclass
    class MapieClassifierParameters:
        mapie_alpha: float = field(default=0.05, metadata=schema(min=0.01))
        test_size: float = field(default=0.1, metadata=schema(min=0.01))
        estimator: MapieClassifierCompatible = field(default=None)
        n_folds: int = field(default=5, metadata=schema(min=2))
        random_state: int = field(default=42, metadata=schema(min=0))

    name: Literal["MapieClassifier"] = "MapieClassifier"
    parameters: MapieClassifierParameters = field(
        default_factory=MapieClassifierParameters
    )

    def estimator(self):
        from optunaz.algorithms.mapie_uncertainty import MapieClassifierWithUncertainty

        return MapieClassifierWithUncertainty(
            mapie_alpha=self.parameters.mapie_alpha,
            test_size=self.parameters.test_size,
            estimator=self.parameters.estimator.estimator(),
            n_jobs=get_n_jobs(self.parameters.estimator),
            n_folds=self.parameters.n_folds,
            random_state=self.parameters.random_state,
        )


AnyAlgorithm = Union[
    AnyUncalibratedClassifier,
    AnyRegression,
    CalibratedClassifierCVWithVA,
    MapieRegressor,
    MapieClassifier,
]


def get_n_jobs(estimator: AnyAlgorithm) -> int:
    """Avoid nested parallelism in estimators."""
    return 1 if isinstance(estimator, AvoidNestedParallelism.__args__) else -1


AnyChemPropAlgorithm = Union[
    ChemPropClassifier,
    ChemPropRegressor,
    ChemPropRegressorPretrained,
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
        n_splits: Optional[int] = field(default=None, metadata=schema(min=1))
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

    def __post_init__(self):
        # Tell scaled descriptor to use the main dataset by default.
        copy_path_for_scaled_descriptor(self.descriptor, self.data)
