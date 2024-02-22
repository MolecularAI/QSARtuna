from PRF import prf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

import numpy as np


class PRFClassifier(BaseEstimator, RegressorMixin):
    """Sklearn-like PRFC classifier"""

    def __init__(
        self,
        n_jobs=-1,
        n_estimators=10,
        criterion="gini",
        max_features="auto",
        use_py_gini=True,
        use_py_leafs=True,
        max_depth=None,
        keep_proba=0.05,
        bootstrap=True,
        new_syn_data_frac=0,
        min_py_sum_leaf=1,
    ):
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.use_py_gini = use_py_gini
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.bootstrap = bootstrap
        self.new_syn_data_frac = new_syn_data_frac
        self.min_py_sum_leaf = min_py_sum_leaf
        self.prf_cls = prf(
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
            use_py_gini=use_py_gini,
            use_py_leafs=use_py_leafs,
            max_depth=max_depth,
            keep_proba=keep_proba,
            bootstrap=bootstrap,
            new_syn_data_frac=new_syn_data_frac,
            min_py_sum_leaf=min_py_sum_leaf,
        )

    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        if sample_weight:
            self.classes_ = unique_labels(y)
            py = np.zeros([len(sample_weight), 2])
            py[:, 1] = sample_weight
            py[:, 0] = 1 - py[:, 1]
            self.prf_cls.fit(X=X.astype(np.float32), py=py.astype(np.float32))
        else:
            self.classes_ = None
            py = np.zeros([len(y), 2])
            py[:, 1] = y
            py[:, 0] = 1 - py[:, 1]
            self.prf_cls.fit(X=X.astype(np.float32), py=py.astype(np.float32))
        return self

    def predict(self, X):
        check_is_fitted(self, ["n_estimators"])
        X = check_array(X)
        if X.shape[0] == 1:
            X = np.concatenate((X, X))
            return np.array(self.prf_cls.predict_proba(X=X.astype(np.float32)))[:1:, 1]
        else:
            return np.array(self.prf_cls.predict_proba(X=X.astype(np.float32)))[:, 1]

    def predict_proba(self, X):
        check_is_fitted(self, ["n_estimators"])
        X = check_array(X)
        if X.shape[0] == 1:
            X = np.concatenate((X, X))
            return np.array(self.prf_cls.predict_proba(X=X.astype(np.float32)))[:1:]
        else:
            return np.array(self.prf_cls.predict_proba(X=X.astype(np.float32)))

    def __getattr__(self, attr):
        return getattr(self.prf_cls, attr)

    def __str__(self):
        sb = []
        do_not_print = [
            "estimators_",
            "label_dict",
            "new_syn_data_frac",
            "prf_cls",
        ]
        for key in self.__dict__:
            if key not in do_not_print:
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        sb = "ProbabilisticRandomForestClassifier(" + ", ".join(sb) + ")"
        return sb
