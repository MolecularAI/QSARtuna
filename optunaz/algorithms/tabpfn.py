import glob
import importlib.util
import io
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
from torch import cuda
from torch.backends import mps


def save_model_memory(model_dir):
    """Saves the model directory as a tarball in memory."""
    tarblob = io.BytesIO()
    with tarfile.TarFile(mode="w", fileobj=tarblob) as tar:
        dirinfo = tarfile.TarInfo(model_dir)
        dirinfo.mode = 0o755
        dirinfo.type = tarfile.DIRTYPE
        tar.addfile(dirinfo, None)
        for dirpath, _, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(dirpath, file)
                with open(file_path, "rb") as fh:
                    filedata = io.BytesIO(fh.read())
                    fileinfo = tarfile.TarInfo(str(file_path))
                    fileinfo.size = len(filedata.getbuffer())
                    tar.addfile(fileinfo, filedata)
    return tarblob


def extract_model_memory(tarblob, temp_dir, save_dir):
    """Extracts the model directory from a tarball in memory."""
    tarblob.seek(0)
    with tarfile.TarFile(mode="r", fileobj=tarblob) as tar:
        for member in tar.getmembers():
            member.name = os.path.relpath(member.name, save_dir)
            tar.extract(member, temp_dir)
    return


@contextmanager
def suppress_logging(level=logging.FATAL):
    """Silence TabPFN outputs"""
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def prep_alg(env_path: str, model_type: str) -> None:
    """Prepare TabPFN if necessary, copying .ckpt files from source"""

    # Use environment variable or default path
    source_dir = os.getenv(
        env_path,
        os.path.expanduser("~/Library/Caches/tabpfn/")
        if sys.platform == "darwin"
        else "/root/.cache/tabpfn/",
    )
    source_files = glob.glob(os.path.join(source_dir, f"tabpfn-v2-{model_type}*.ckpt"))

    # Dynamically find the location of the tabpfn_extensions package
    spec = importlib.util.find_spec("tabpfn_extensions")
    if not spec or not spec.submodule_search_locations:
        raise ImportError("tabpfn_extensions package not found")
    dest_dir = spec.submodule_search_locations[0]

    # Copy each model file to the destination
    for file in source_files:
        dest_path = os.path.join(dest_dir, "hpo", "hpo_models", os.path.basename(file))
        if not os.path.exists(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file, dest_path)
    return


def set_device(mps_fallback: str = "cpu") -> str:
    """Set the device to ensure that resources are available across platforms. MPS is not currently supported"""
    return (
        mps_fallback if mps.is_available() else "cuda" if cuda.is_available() else "cpu"
    )


class BaseTabPFN(BaseEstimator):
    """Base class for TabPFN models with common functionality"""

    def __init__(
        self,
        max_time: int = 30,
        random_state: int | np.random.RandomState = 42,
        max_feats: int = 500,
        feature_selection: Literal["k_best", "tree"] = "k_best",
        eval_metric: str | None = None,
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.max_feats = max_feats
        self.feature_selection = feature_selection
        self.eval_metric = eval_metric

    def fit(self, X, y):
        X = check_array(X, ensure_2d=True, allow_nd=False)
        y = check_array(y, ensure_2d=False, allow_nd=False)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]

        model_class, prep_path, feat_alg, score_func = (
            (
                AutoTabPFNClassifier,
                "TABPFN_CLS__PATH",
                RandomForestClassifier,
                f_classif,
            )
            if is_classifier(self)
            else (
                AutoTabPFNRegressor,
                "TABPFN_REG__PATH",
                RandomForestRegressor,
                f_regression,
            )
        )
        ignore_limits = len(y) >= 10000
        phe_init_args = {"SUBSAMPLE_SAMPLES": 10000} if ignore_limits else {}

        with tempfile.TemporaryDirectory() as output_dir:
            phe_init_args["path"] = output_dir
            self.output_dir_ = output_dir

            self.model_ = model_class(
                max_time=self.max_time,
                random_state=self.random_state,
                eval_metric=self.eval_metric,
                ignore_pretraining_limits=ignore_limits,
                phe_init_args=phe_init_args,
            )

            prep_alg(prep_path, "classifier" if is_classifier(self) else "regressor")

            if is_classifier(self):
                self.classes_ = unique_labels(y).astype(np.uint8)

            if X.shape[1] > self.max_feats:
                self._selector = (
                    SelectKBest(score_func=score_func, k=self.max_feats).fit(X, y)
                    if self.feature_selection == "k_best"
                    else feat_alg(random_state=self.random_state).fit(X, y)
                )
                self._important_feats = (
                    self._selector.get_support(indices=True)
                    if self.feature_selection == "k_best"
                    else self._selector.feature_importances_.argsort()[
                        -self.max_feats :
                    ]
                )
                X = X[:, self._important_feats]

            self.model_.device = set_device()
            self.model_.ignore_pretraining_limits = (
                self.model_.ignore_pretraining_limits or self.max_feats > 500
            )
            with suppress_logging():
                self.model_.fit(X, y)
                self.model_tar_ = save_model_memory(output_dir)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_"])
        X = check_array(X, ensure_2d=True, allow_nd=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]} features."
            )
        if hasattr(self, "_important_feats"):
            X = X[:, self._important_feats]
        self.model_.device = set_device()
        with suppress_logging():
            with tempfile.TemporaryDirectory() as tmpdir:
                extract_model_memory(self.model_tar_, tmpdir, self.output_dir_)
                self.model_.predictor_ = TabularPredictor.load(tmpdir)
                preds = (
                    self.model_.predict_proba(X)
                    if is_classifier(self)
                    else self.model_.predict(X)
                )
        return preds

    def predict(self, X):
        if is_classifier(self):
            return self.predict_proba(X)[:, 1] > 0.5

        predictions = self.predict_proba(X).flatten()
        # clip probabilistic predictions
        return (
            predictions.clip(0, 1)
            if 0 <= self.y_.min() <= 1 and 0 <= self.y_.max() <= 1
            else predictions
        )

    def _predict_uncert(self, X):
        """Predict uncertainty for regression or classification tasks."""
        check_is_fitted(self, ["model_"])
        X = check_array(X, ensure_2d=True, allow_nd=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]} features."
            )
        if hasattr(self, "_important_feats"):
            X = X[:, self._important_feats]
        self.model_.device = set_device()
        with suppress_logging():
            with tempfile.TemporaryDirectory() as tmpdir:
                extract_model_memory(self.model_tar_, tmpdir, self.output_dir_)
                self.model_.predictor_ = TabularPredictor.load(tmpdir)
                model_names = self.model_.predictor_.model_names()
                predictor = self.model_.predictor_
                preds = []
                cols = [f"f{i}" for i in range(X.shape[1])]
                pred_func = (
                    predictor.predict_proba
                    if is_classifier(self)
                    else predictor.predict
                )
                for model_name in model_names:
                    if is_classifier(self):
                        preds.append(
                            pred_func(pd.DataFrame(X, columns=cols), model=model_name)
                        )
                    else:
                        preds.append(
                            predictor.predict(
                                pd.DataFrame(X, columns=cols), model=model_name
                            )
                        )
        uncerts = preds.std(axis=0)
        return np.std(uncerts, axis=0)


class TabPFNRegressor(RegressorMixin, BaseTabPFN):
    """Sklearn-like TabPFN Regressor"""

    def __init__(
        self,
        max_time: int = 30,
        random_state: int | np.random.RandomState = 42,
        max_feats: int = 500,
        feature_selection: Literal["k_best", "tree"] = "k_best",
        eval_metric: Literal[
            "root_mean_squared_error", "mse", "mae"
        ] = "root_mean_squared_error",
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.max_feats = max_feats
        self.feature_selection = feature_selection
        self.eval_metric = eval_metric


class TabPFNClassifier(ClassifierMixin, BaseTabPFN):
    """Sklearn-like TabPFN Classifier"""

    def __init__(
        self,
        max_time: int = 30,
        random_state: int | np.random.RandomState = 42,
        max_feats: int = 500,
        feature_selection: Literal["k_best", "tree"] = "k_best",
        eval_metric: Literal["accuracy", "roc_auc", "f1", "log_loss"] = "accuracy",
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.max_feats = max_feats
        self.feature_selection = feature_selection
        self.eval_metric = eval_metric
