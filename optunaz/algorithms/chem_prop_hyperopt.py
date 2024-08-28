import sys
import tarfile
import io
import warnings
from io import StringIO
import os
import logging.config
import chemprop
import torch
from functools import partial
from chemprop.data.utils import get_invalid_smiles_from_list
from chemprop.data import MoleculeDataLoader
from chemprop.interpret import interpret
import pandas as pd
import tempfile
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from joblib import effective_n_jobs
from optunaz.algorithms.side_info import binarise_side_info, process_side_info

logging.getLogger("train").setLevel(logging.ERROR)
logging.getLogger("train").disabled = True
logging.getLogger("train").propagate = False
os.environ["TQDM_DISABLE"] = "1"

np.seterr(divide="ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Monkey patch MoleculeDataLoader num_workers=0 for parallel interpret. num_workers is baked in at 8 and
# setting to 0 avoids unintended nested parallelization, since we already parallelize self.interpret
_MoleculeDataLoader = MoleculeDataLoader
MoleculeDataLoader = partial(_MoleculeDataLoader, num_workers=0)
chemprop.interpret.MoleculeDataLoader = MoleculeDataLoader


class CaptureStdOut(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout


def save_model_memory(model_dir):
    tarblob = io.BytesIO()
    tar = tarfile.TarFile(mode="w", fileobj=tarblob)
    dirinfo = tarfile.TarInfo(model_dir)
    dirinfo.mode = 0o755
    dirinfo.type = tarfile.DIRTYPE
    tar.addfile(dirinfo, None)
    for dirpath, _, files in os.walk(model_dir):
        for file in files:
            with open(os.path.join(dirpath, file), "rb") as fh:
                filedata = io.BytesIO(fh.read())
                fileinfo = tarfile.TarInfo(os.path.join(dirpath, file))
                fileinfo.size = len(filedata.getbuffer())
                tar.addfile(fileinfo, filedata)
    tar.close()
    return tarblob


def extract_model_memory(tarblob, temp_dir, save_dir):
    tarblob.seek(0)
    tar = tarfile.TarFile(mode="r", fileobj=tarblob)
    for member in tar.getmembers():
        member.name = os.path.relpath(member.name, save_dir)
        tar.extract(member, temp_dir)  # extract
    tar.close()
    return


def get_search_parameter_keywords(level):
    search_parameter_keywords_dict = {
        "1": ["basic"],
        "2": ["basic", "linked_hidden_size"],
        "3": ["basic", "linked_hidden_size", "learning_rate"],
        "4": ["basic", "hidden_size", "ffn_hidden_size", "learning_rate"],
        "5": ["basic", "hidden_size", "ffn_hidden_size", "learning_rate", "activation"],
        "6": [
            "basic",
            "hidden_size",
            "ffn_hidden_size",
            "learning_rate",
            "activation",
            "batch_size",
        ],
        "7": [
            "basic",
            "hidden_size",
            "ffn_hidden_size",
            "learning_rate",
            "activation",
            "batch_size",
            "aggregation_norm",
        ],
        "8": ["all"],
    }
    return search_parameter_keywords_dict[level]


def get_search_parameter_level(len_y, features_generator, num_iters, epochs):
    level = pd.cut(
        [len_y],
        bins=[0, 50, 150, 250, 500, 1000, 5000, 99999999],
        labels=["8", "6", "5", "4", "3", "2", "1"],
        right=False,
    )[0]
    if features_generator != "none":
        level = str(min(int(level), 6))
    if num_iters <= 200:
        level = str(min(int(level), 5))
    if num_iters < 150 or features_generator == "rdkit_2d_normalized":
        level = str(min(int(level), 4))
    if (
        num_iters < 100
        or epochs > 250
        or (features_generator == "rdkit_2d_normalized" and len_y > 2500)
    ):
        level = str(min(int(level), 3))
    if (
        num_iters < 50
        or epochs >= 300
        or (features_generator != "none" and num_iters < 100)
    ):
        level = str(min(int(level), 2))
    if (
        (num_iters < 30)
        or epochs >= 350
        or (features_generator != "none" and num_iters < 50)
    ):
        level = str(min(int(level), 1))
    return level


class BaseChemPropHyperopt(BaseEstimator):
    """Scikit-learn-like Chemprop w/ hyperopt & w/ multi-task functionality"""

    def __init__(
        self,
        aux_weight_pc=100,
        dataset_type=None,
        ensemble_size=1,
        epochs=30,
        features_generator="none",  # none, morgan, morgan_count, rdkit_2d, rdkit_2d_normalized
        num_iters=1,
        num_workers=-1,
        search_parameter_level="auto",
        seed=0,
        side_info_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
    ):
        self.aux_weight_pc = aux_weight_pc
        self.dataset_type = dataset_type
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.features_generator = features_generator
        self.num_workers = num_workers
        self.num_iters = num_iters
        self.search_parameter_level = search_parameter_level
        self.side_info_rfe = side_info_rfe
        self.seed = seed
        self.split_sizes = split_sizes
        self.num_workers = effective_n_jobs(num_workers)

    def fit(self, X, y):
        X = np.array(X)
        side_info = None
        X_aux = None
        if len(X.shape) == 1:
            X = np.array(X).reshape(len(X), 1)
        else:
            side_info = np.concatenate(X[:, 1])
            if (side_info == None).all():
                side_info = None
            if X.shape[1] > 2:
                X_aux = X[:, 2:]
            X = np.array(X[:, 0].reshape(len(X), 1))
        self.x_aux_ = X_aux
        self.side_info_ = side_info
        self.X_ = X

        y = np.array(y)
        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)

        if self.dataset_type == "classification":
            self.classes_ = unique_labels(y).astype(np.uint8)
            self._estimator_type = "classifier"
            if self.side_info_ is not None:
                try:
                    _ = unique_labels(self.side_info_)
                except ValueError:
                    self.side_info_ = binarise_side_info(self.side_info_)
            y = y.astype(np.uint8)
        elif self.dataset_type == "regression":
            self._estimator_type = "regressor"

        if self.side_info_ is not None:
            if self.side_info_rfe:
                si = process_side_info(self.side_info_, y=y)
            else:
                si = process_side_info(self.side_info_)
            y = np.hstack((y, si))
        self.y_ = y

        if self.search_parameter_level == "auto":
            self.search_parameter_level = get_search_parameter_level(
                len(self.y_), self.features_generator, self.num_iters, self.epochs
            )
        self.search_parameter_keywords = get_search_parameter_keywords(
            self.search_parameter_level
        )
        self.target_columns = list(range(y.shape[1]))

        with tempfile.TemporaryDirectory() as save_dir:
            self.save_dir = save_dir
            with tempfile.NamedTemporaryFile(
                delete=True, mode="w+", suffix=".csv"
            ) as data_path:
                hyp_arguments = [
                    "--num_iters",
                    str(int(self.num_iters)),
                    "--search_parameter_keywords",
                ] + list(map(str, self.search_parameter_keywords))

                train_arguments = [
                    "--save_dir",
                    f"{save_dir}",
                    "--ensemble_size",
                    f"{int(self.ensemble_size)}",
                    "--split_sizes",
                ] + list(map(str, self.split_sizes))

                for arguments in [hyp_arguments, train_arguments]:
                    arguments += [
                        "--data_path",
                        f"{data_path.name}",
                        "--log_frequency",
                        "99999",
                        "--cache_cutoff",
                        "inf",
                        "--quiet",
                        "--epochs",
                        str(int(self.epochs)),
                        "--num_workers",
                        str(int(self.num_workers)),
                        "--dataset_type",
                        self.dataset_type,
                        "--seed",
                        str(self.seed),
                        "--pytorch_seed",
                        str(self.seed),
                        "--target_columns",
                    ] + list(map(str, self.target_columns))
                    if not torch.cuda.is_available():
                        arguments += ["--no_cuda"]
                    else:
                        arguments += ["--gpu", "0"]
                    weights_ = [
                        100 if targ == 0 else self.aux_weight_pc
                        for targ in self.target_columns
                    ]
                    self.target_weights = weights_
                    arguments += ["--target_weights"] + list(map(str, weights_))
                    if self.dataset_type == "classification":
                        arguments += ["--class_balance"]
                    if self.features_generator != "none":
                        arguments += [
                            "--features_generator",
                            f"{self.features_generator}",
                        ]
                        if self.features_generator == "rdkit_2d_normalized":
                            arguments += ["--no_features_scaling"]  # already pre-scaled

                pd.DataFrame(
                    np.hstack((X, y)),
                    columns=["Smiles"] + list(map(str, range(y.shape[1]))),
                ).to_csv(data_path.name, index=False)

                if self.x_aux_ is not None:
                    x_aux_path = tempfile.NamedTemporaryFile(
                        delete=True, mode="w+", suffix=".csv"
                    )
                    pd.DataFrame(
                        self.x_aux_,
                    ).to_csv(x_aux_path.name, index=False)
                    # arguments += ["--features_path", f"{x_aux_path.name}"] TODO: allow features once ChemProp is updated

                with CaptureStdOut() as _:
                    if self.num_iters > 1:
                        with tempfile.NamedTemporaryFile(
                            delete=True, mode="w+"
                        ) as config_save_path:
                            hyp_arguments += [
                                "--config_save_path",
                                f"{config_save_path.name}",
                            ]
                            train_arguments += ["--config_path", f"{config_save_path.name}"]
                            hyp_args = chemprop.args.HyperoptArgs().parse_args(
                                hyp_arguments
                            )
                            chemprop.hyperparameter_optimization.hyperopt(args=hyp_args)
                            train_args = chemprop.args.TrainArgs().parse_args(
                                train_arguments
                            )
                            chemprop.train.cross_validate(
                                args=train_args, train_func=chemprop.train.run_training
                            )
                    else:
                        train_args = chemprop.args.TrainArgs().parse_args(train_arguments)
                        chemprop.train.cross_validate(
                            args=train_args, train_func=chemprop.train.run_training
                        )
                self.model_ = save_model_memory(save_dir)
                for pre_param, pre_value in train_args.__dict__.items():
                    if not hasattr(self, pre_param):
                        if not isinstance(pre_value, (str, bool, int, float)):
                            continue
                        if not (
                            pre_param
                            in [
                                "argument_buffer",
                                "class_variables",
                                "description",
                                "argument_default",
                            ]
                            and not pre_param[0] == "_"
                        ):
                            self.__dict__[pre_param] = pre_value
        if self.x_aux_ is not None:
            x_aux_path.close()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_"])
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            arguments = [
                "--test_path",
                "/dev/null",
                "--preds_path",
                "/dev/null",
                "--checkpoint_dir",
                f"{checkpoint_dir}",
                "--num_workers",
                "0",  # >0 causes this issue https://github.com/facebookresearch/hydra/issues/964
            ]
            if not torch.cuda.is_available():
                arguments += ["--no_cuda"]
            else:
                arguments += ["--gpu"]
                arguments += ["0"]
            if self.features_generator != "none":
                arguments += ["--features_generator", f"{self.features_generator}"]
                if self.features_generator == "rdkit_2d_normalized":
                    arguments += ["--no_features_scaling"]  # already pre-scaled
            extract_model_memory(self.model_, checkpoint_dir, self.save_dir)

            X = np.array(X)
            if self.x_aux_ is not None:
                x_aux_path = tempfile.NamedTemporaryFile(
                    delete=True, mode="w+", suffix=".csv"
                )
                pd.DataFrame(X[:, 2:]).to_csv(x_aux_path.name, index=False)
                # arguments += ["--features_path", f"{x_aux_path.name}"] TODO: allow features once ChemProp is updated
            if len(X.shape) == 1:
                X = np.array(X).reshape(len(X), 1)
            else:
                X = np.array(X[:, 0].reshape(len(X), 1))

            with CaptureStdOut() as _:
                args = chemprop.args.PredictArgs().parse_args(arguments)
                model_objects = chemprop.train.load_model(args=args)
                preds = np.array(
                    chemprop.train.make_predictions(
                        args=args, model_objects=model_objects, smiles=X
                    ),
                    dtype="<U32",
                )
            if self.x_aux_ is not None:
                x_aux_path.close()
        preds[preds == "Invalid SMILES"] = np.nan
        preds = preds.astype(np.float32)
        if self.dataset_type == "classification":
            yhats = np.zeros([len(preds), 2])
            yhats[:, 1] = preds[:, 0]
            yhats[:, 0] = 1 - yhats[:, 1]
            return yhats
        else:
            if preds.shape[1] != 1:
                preds = preds[:, 0].reshape(len(X), 1)
            return preds

    def predict(self, X):
        if self.dataset_type == "classification":
            return self.predict_proba(X)[:, 1] > 0.5
        else:
            # clip probabilistic predictions
            if 0 <= self.y_.min() <= 1 and 0 <= self.y_.max() <= 1:
                return self.predict_proba(X).clip(0, 1)
            else:
                return self.predict_proba(X)

    def predict_uncert(self, X):
        check_is_fitted(self, ["model_"])
        if self.ensemble_size > 1:
            uncertainty_method = "ensemble"
        else:
            uncertainty_method = "dropout"
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            arguments = [
                "--test_path",
                "/dev/null",
                "--preds_path",
                "/dev/null",
                "--checkpoint_dir",
                f"{checkpoint_dir}",
                "--num_workers",
                "0",  # >0 causes this issue https://github.com/facebookresearch/hydra/issues/964
                "--uncertainty_method",
                f"{uncertainty_method}",
            ]
            if not torch.cuda.is_available():
                arguments += ["--no_cuda"]
            else:
                arguments += ["--gpu"]
                arguments += ["0"]
            if self.features_generator != "none":
                arguments += ["--features_generator", f"{self.features_generator}"]
                if self.features_generator == "rdkit_2d_normalized":
                    arguments += ["--no_features_scaling"]  # already pre-scaled
            extract_model_memory(self.model_, checkpoint_dir, self.save_dir)

            X = np.array(X)
            if self.x_aux_ is not None:
                x_aux_path = tempfile.NamedTemporaryFile(
                    delete=True, mode="w+", suffix=".csv"
                )
                pd.DataFrame(X[:, 2:]).to_csv(x_aux_path.name, index=False)
                # arguments += ["--features_path", f"{x_aux_path.name}"] TODO: allow features once ChemProp is updated
            if len(X.shape) == 1:
                X = np.array(X).reshape(len(X), 1)
            else:
                X = np.array(X[:, 0].reshape(len(X), 1))

            with CaptureStdOut() as _:
                args = chemprop.args.PredictArgs().parse_args(arguments)
                if uncertainty_method == "dropout":
                    model_objects = list(chemprop.train.load_model(args=args))
                    model_objects[3] = iter(model_objects[3])
                    model_objects[2] = iter(model_objects[2])
                    model_objects = tuple(model_objects)
                else:
                    model_objects = chemprop.train.load_model(args=args)
                preds = np.array(
                    chemprop.train.make_predictions(
                        args=args,
                        model_objects=model_objects,
                        smiles=X,
                        return_uncertainty=True,
                    ),
                    dtype="<U32",
                )
            if self.x_aux_ is not None:
                x_aux_path.close()
        preds[preds == "Invalid SMILES"] = np.nan
        preds[preds == "None"] = np.nan
        for p in range(preds.shape[0]):
            preds[p] = preds[p].astype(np.float32)
            if preds[p].shape[1] != 1:
                preds[p][:, 0].reshape(len(X), 1)
        return preds

    def interpret(self, X, prop_delta=0.75):
        check_is_fitted(self, ["model_"])
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            extract_model_memory(self.model_, checkpoint_dir, self.save_dir)
            with tempfile.NamedTemporaryFile(
                delete=True, mode="r+", suffix=".csv"
            ) as data_path:
                intrprt_args = [
                    "--data_path",
                    data_path.name,
                    "--checkpoint_dir",
                    checkpoint_dir,
                    "--property_id",
                    "1",
                    "--prop_delta",
                    f"{prop_delta}",
                    "--num_workers",
                    "0",  # paralellize self.interpret only and avoid nesting with parallelized num_workers
                ]

                X = np.array(X)
                if self.x_aux_ is not None:
                    x_aux_path = tempfile.NamedTemporaryFile(
                        delete=True, mode="w+", suffix=".csv"
                    )
                    pd.DataFrame(X[:, 2:]).to_csv(x_aux_path.name, index=False)
                    # intrprt_args += ["--features_path", f"{x_aux_path.name}"] TODO: allow features once ChemProp is updated
                if len(X.shape) == 1:
                    X = np.array(X).reshape(len(X), 1)
                else:
                    X = np.array(X[:, 0].reshape(len(X), 1))
                X = pd.DataFrame(X, columns=["smiles"])
                X.to_csv(data_path.name, index=False)
                args = chemprop.args.InterpretArgs().parse_args(intrprt_args)
                with CaptureStdOut() as intrprt:
                    interpret(args=args)
        intrprt = [
            line.split(",")
            for line in intrprt
            if not any([load_elapse in line for load_elapse in ["Loading", "Elapsed"]])
        ]
        intrprt = pd.DataFrame(np.vstack(intrprt[1:]), columns=intrprt[0])
        intrprt.smiles = intrprt.smiles.str[2:-2]
        return X.merge(intrprt, on="smiles", how="outer")

    def chemprop_fingerprint(self, X, fingerprint_type="MPN"):
        """Loads the trained model and uses it to encode fingerprint vectors for the data.
        Fingerprints are returned to the user as pd.DataFrame, with shape, ncompds vs. nlatent representation vectors
        """
        check_is_fitted(self, ["model_"])
        X = np.array(X)
        if self.x_aux_ is not None:
            x_aux_path = tempfile.NamedTemporaryFile(
                delete=True, mode="w+", suffix=".csv"
            )
            pd.DataFrame(X[:, 2:]).to_csv(x_aux_path.name, index=False)

        if len(X.shape) == 1:
            X = np.array(X).reshape(len(X), 1)
        else:
            X = np.array(X[:, 0].reshape(len(X), 1))

        invalid_smiles = get_invalid_smiles_from_list(X)
        valid_idx = [idx for idx, smi in enumerate(X) if smi not in invalid_smiles]
        valid_smiles = [smi for smi in X if smi not in invalid_smiles]
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            extract_model_memory(self.model_, checkpoint_dir, self.save_dir)
            fprnt_args = [
                "--test_path",
                "/dev/null",
                "--preds_path",
                "/dev/null",
                "--checkpoint_dir",
                checkpoint_dir,
                "--num_workers",
                "0",
                "--fingerprint_type",
                f"{fingerprint_type}",
            ]
            # if self.x_aux_ is not None:
            # fprnt_args += ["--features_path", f"{x_aux_path.name}"] TODO: allow features once ChemProp is updated
            # load_model returns pred&train arguments, object models & tasks info - but we only need TrainArgs here
            with CaptureStdOut() as _:
                args = chemprop.args.FingerprintArgs().parse_args(fprnt_args)
                _, trainargs, _, _, _, _ = chemprop.train.load_model(args=args)
                if fingerprint_type == "MPN":
                    numpy_fp = np.zeros((len(X), trainargs.hidden_size))
                elif fingerprint_type == "last_FFN":
                    numpy_fp = np.zeros((len(X), trainargs.ffn_hidden_size))
                else:
                    raise ValueError("fingerprint_type should be one of ['MPN','last_FFN']")
                numpy_fp[:] = np.nan
                try:
                    fps = chemprop.train.molecule_fingerprint.molecule_fingerprint(
                        args=args, smiles=valid_smiles
                    )
                    fps = fps.reshape(fps.shape[:-1])
                    numpy_fp[valid_idx] = fps
                except (ValueError, AttributeError):
                    pass
            if self.x_aux_ is not None:
                x_aux_path.close()
        return pd.DataFrame(numpy_fp)

    def __str__(self):
        sb = []
        do_not_print = [
            "X_",
            "y_",
            "x_aux_",
            "num_workers",
            "hash_",
            "model_",
            "train_args",
            "hyp_args",
            "side_info_",
            "target_columns",
            "target_weight",
        ]
        for key in self.__dict__:
            if key not in do_not_print:
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        sb = "ChemPropHyperopt(" + ", ".join(sb) + ")"
        return sb


class ChemPropHyperoptClassifier(ClassifierMixin, BaseChemPropHyperopt):
    def __init__(
        self,
        aux_weight_pc=100,
        dataset_type="classification",
        ensemble_size=1,
        epochs=30,
        features_generator="none",  # none, morgan, morgan_count, rdkit_2d, rdkit_2d_normalized
        num_iters=1,
        num_workers=-1,
        search_parameter_level="auto",
        seed=0,
        side_info_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
    ):
        self.aux_weight_pc = aux_weight_pc
        self.dataset_type = dataset_type
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.features_generator = features_generator
        self.num_workers = num_workers
        self.num_iters = num_iters
        self.search_parameter_level = search_parameter_level
        self.side_info_rfe = side_info_rfe
        self.seed = seed
        self.split_sizes = split_sizes
        self.num_workers = effective_n_jobs(num_workers)


class ChemPropHyperoptRegressor(RegressorMixin, BaseChemPropHyperopt):
    def __init__(
        self,
        aux_weight_pc=100,
        dataset_type="regression",
        ensemble_size=1,
        epochs=30,
        features_generator="none",  # none, morgan, morgan_count, rdkit_2d, rdkit_2d_normalized
        num_iters=1,
        num_workers=-1,
        search_parameter_level="auto",
        seed=0,
        side_info_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
    ):
        self.aux_weight_pc = aux_weight_pc
        self.dataset_type = dataset_type
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.features_generator = features_generator
        self.num_workers = num_workers
        self.num_iters = num_iters
        self.search_parameter_level = search_parameter_level
        self.side_info_rfe = side_info_rfe
        self.seed = seed
        self.split_sizes = split_sizes
        self.num_workers = effective_n_jobs(num_workers)
