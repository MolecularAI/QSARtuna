import copy
import tempfile
import dill
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class CustomCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Customised implementation of catboost classifier which allows for deepcopy and quiet training

    The explainability code requires that a deep copy of each estimator is possible and
    without the additional logic in CustomCatBoostClassifier, deserialisation of the
    base catboost algorithms occurs, which removes the customised attributes added by Qptuna
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.classifier = None

    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        with tempfile.TemporaryDirectory() as train_dir:
            self.classifier = CatBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_strength=self.random_strength,
                random_state=42,
                verbose=False,
                train_dir=train_dir
            )
            self.classifier.fit(X, y, sample_weight=sample_weight, verbose=False)
        return self

    def predict(self, X):
        return (self.classifier.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self, ["X_"])
        return self.classifier.predict_proba(X)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != "classifier":
                setattr(result, k, copy.deepcopy(v, memo))

        clf = getattr(self, "classifier", None)
        if clf is not None:
            result.classifier = dill.loads(dill.dumps(clf))
        else:
            result.classifier = None

        return result

    def __getattr__(self, attr):
        return getattr(self.classifier, attr)

    def __str__(self):
        sb = []
        do_not_print = [
            "X_",
            "classifier",
        ]
        for key in self.__dict__:
            if key not in do_not_print:
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        sb = "CustomCatBoostClassifier(" + ", ".join(sb) + ")"
        return sb


class CustomCatBoostRegressor(RegressorMixin, BaseEstimator):
    """Customised implementation of catboost regressor which allows for deepcopy and quiet training

    The explainability code requires that a deep copy of each estimator is possible and
    without the additional logic in CustomCatBoostClassifier, deserialisation of the
    base catboost algorithms occurs, which removes the customised attributes added by Qptuna
    """
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.regressor = None

    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        with tempfile.TemporaryDirectory() as train_dir:
            self.regressor = CatBoostRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_strength=self.random_strength,
                random_state=42,
                verbose=False,
                train_dir=train_dir
            )
            self.regressor.fit(X, y, sample_weight=sample_weight, verbose=False)
        return self

    def predict(self, X):
        check_is_fitted(self, ["X_"])
        return self.regressor.predict(X)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != "regressor":
                setattr(result, k, copy.deepcopy(v, memo))

        clf = getattr(self, "regressor", None)
        if clf is not None:
            result.regressor = dill.loads(dill.dumps(clf))
        else:
            result.regressor = None

        return result

    def __getattr__(self, attr):
        return getattr(self.regressor, attr)

    def __str__(self):
        sb = []
        do_not_print = [
            "X_",
            "regressor",
        ]
        for key in self.__dict__:
            if key not in do_not_print:
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        sb = "CustomCatBoostRegressor(" + ", ".join(sb) + ")"
        return sb