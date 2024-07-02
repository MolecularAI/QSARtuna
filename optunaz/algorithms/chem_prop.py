import sys
import tarfile
import io
from io import StringIO
import os
import logging.config
import math
import chemprop
import torch
import pickle
from functools import partialmethod, partial
from chemprop.data.utils import get_invalid_smiles_from_list
from chemprop.data import MoleculeDataLoader
from chemprop.interpret import interpret
from chemprop import args
import pandas as pd
import tempfile
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from joblib import effective_n_jobs
from optunaz.algorithms.side_info import binarise_side_info, process_side_info
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
logging.getLogger("train").disabled = True
logging.getLogger("train").propagate = False
logging.getLogger("train").setLevel(logging.ERROR)
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
np.seterr(divide="ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Monkey patch MoleculeDataLoader num_workers=0 for parallel interpret. num_workers is baked in at 8 and
# setting to 0 avoids unintended nested parallelization, since we already parallelize self.interpret
_MoleculeDataLoader = MoleculeDataLoader
MoleculeDataLoader = partial(_MoleculeDataLoader, num_workers=0)
chemprop.interpret.MoleculeDataLoader = MoleculeDataLoader


class QSARtunaCommonArgs(args.CommonArgs):
    def __init__(self, *args, **kwargs):
        super(QSARtunaCommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0
        self._bond_descriptors_size = 0
        self._atom_constraints = []
        self._bond_constraints = []

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if self.cuda:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cuda", self.gpu)
        else:
            return torch.device("cpu")

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda" or "mps"
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs), MPS, or not."""
        return not self.no_cuda and (
            torch.cuda.is_available() or torch.backends.mps.is_available()
        )


class QSARtunaTrainArgs(args.TrainArgs, QSARtunaCommonArgs):
    def __init__(self, *args, **kwargs) -> None:
        super(QSARtunaTrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None


class QSARtunaPredictArgs(args.PredictArgs, QSARtunaCommonArgs):
    pass


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


class BaseChemProp(BaseEstimator):
    """
    Scikit-learn-like Chemprop w/o hyperopt & w/ multitask functionality
    """

    def __init__(
        self,
        activation="ReLU",  # ReLU, LeakyReLU, PReLU, tanh, SELU, ELU
        aggregation="mean",  # mean, sum, norm
        aggregation_norm=100,
        aux_weight_pc=100,
        batch_size=50,
        dataset_type=None,
        depth=3,
        dropout=0.0,
        ensemble_size=1,
        epochs=30,
        ffn_hidden_size=300,
        ffn_num_layers=2,
        final_lr_ratio_exp=-4,
        hidden_size=300,
        init_lr_ratio_exp=-4,
        max_lr_exp=-3,
        num_workers=-1,
        seed=0,
        side_info_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
        warmup_epochs_ratio=0.1,
        features_generator="none",  # none, morgan, morgan_count, rdkit_2d, rdkit_2d_normalized
    ):
        self.activation = activation
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm
        self.aux_weight_pc = aux_weight_pc
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.depth = depth
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.features_generator = features_generator
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn_num_layers = ffn_num_layers
        self.final_lr_ratio_exp = final_lr_ratio_exp
        self.hidden_size = hidden_size
        self.init_lr_ratio_exp = init_lr_ratio_exp
        self.max_lr_exp = max_lr_exp
        self.num_workers = effective_n_jobs(num_workers)
        self.seed = seed
        self.split_sizes = split_sizes
        self.side_info_rfe = side_info_rfe
        self.warmup_epochs_ratio = warmup_epochs_ratio
        self.max_lr = 10**self.max_lr_exp
        self.init_lr = (10**self.init_lr_ratio_exp) * self.max_lr
        self.final_lr = (10**self.final_lr_ratio_exp) * self.max_lr

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
        self.target_columns = list(range(y.shape[1]))

        with tempfile.TemporaryDirectory() as save_dir:
            self.save_dir = save_dir
            with tempfile.NamedTemporaryFile(delete=True, mode="w+") as data_path:
                arguments = (
                    [
                        "--data_path",
                        f"{data_path.name}",
                        "--dataset_type",
                        self.dataset_type,
                        "--save_dir",
                        f"{save_dir}",
                        "--log_frequency",
                        "99999",
                        "--cache_cutoff",
                        "inf",
                        "--quiet",
                        "--seed",
                        self.seed,
                        "--pytorch_seed",
                        self.seed,
                        "--target_columns",
                    ]
                    + list(map(str, self.target_columns))
                    + [
                        "--activation",
                        self.activation,
                        "--aggregation",
                        self.aggregation,
                        "--aggregation_norm",
                        str(self.aggregation_norm),
                        "--batch_size",
                        str(self.batch_size),
                        "--depth",
                        str(int(self.depth)),
                        "--dropout",
                        str(int(self.dropout)),
                        "--ensemble_size",
                        str(int(self.ensemble_size)),
                        "--epochs",
                        str(int(self.epochs)),
                        "--ffn_hidden_size",
                        str(int(self.ffn_hidden_size)),
                        "--ffn_num_layers",
                        str(int(self.ffn_num_layers)),
                        "--final_lr",
                        str(self.final_lr),
                        "--hidden_size",
                        str(int(self.hidden_size)),
                        "--init_lr",
                        str(self.init_lr),
                        "--num_workers",
                        str(int(self.num_workers)),
                        "--max_lr",
                        str(self.max_lr),
                        "--split_sizes",
                    ]
                    + list(map(str, self.split_sizes))
                )
                if (
                    not torch.cuda.is_available()
                    and not torch.backends.mps.is_available()
                ):
                    arguments += ["--no_cuda"]
                elif torch.cuda.is_available():
                    arguments += ["--gpu", "0"]
                if self.dataset_type == "classification":
                    arguments += ["--class_balance"]
                weights_ = [
                    100 if targ == 0 else self.aux_weight_pc
                    for targ in self.target_columns
                ]
                self.target_weights = weights_
                arguments += ["--target_weights"] + list(map(str, weights_))
                if self.features_generator != "none":
                    arguments += ["--features_generator", f"{self.features_generator}"]
                    if self.features_generator == "rdkit_2d_normalized":
                        arguments += ["--no_features_scaling"]  # already pre-scaled

                pd.DataFrame(
                    np.hstack((self.X_, self.y_)),
                    columns=["Smiles"] + list(map(str, range(self.y_.shape[1]))),
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
                    args = QSARtunaTrainArgs().parse_args(arguments)
                    chemprop.train.cross_validate(
                        args=args, train_func=chemprop.train.run_training
                    )
            self.model_ = save_model_memory(save_dir)
        if self.x_aux_ is not None:
            x_aux_path.close()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_"])
        if self.num_workers > effective_n_jobs(-1):
            num_workers = effective_n_jobs(-1)
        else:
            num_workers = self.num_workers
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            arguments = [
                "--test_path",
                "/dev/null",
                "--preds_path",
                "/dev/null",
                "--checkpoint_dir",
                f"{checkpoint_dir}",
                "--num_workers",
                f"{num_workers}",
            ]
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                arguments += ["--no_cuda"]
            elif torch.cuda.is_available():
                arguments += ["--gpu", "0"]
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
            if len(X.shape) > 1:
                X = np.array(X[:, 0].reshape(len(X), 1))
            else:
                X = np.array(X).reshape(len(X), 1)

            with CaptureStdOut() as _:
                args = QSARtunaPredictArgs().parse_args(arguments)
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
        if self.num_workers > effective_n_jobs(-1):
            num_workers = effective_n_jobs(-1)
        else:
            num_workers = self.num_workers
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
                f"{num_workers}",
                "--uncertainty_method",
                f"{uncertainty_method}",
            ]
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                arguments += ["--no_cuda"]
            elif torch.cuda.is_available():
                arguments += ["--gpu", "0"]
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
                args = QSARtunaPredictArgs().parse_args(arguments)
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
        if preds[0].shape[1] != 1:
            return [p[:, 0].reshape(len(X), 1).astype(np.float32) for p in preds]
        else:
            return preds.astype(np.float32)

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
                    # arguments += ["--features_path", f"{x_aux_path.name}"] TODO: allow features once ChemProp is updated
                if len(X.shape) == 1:
                    X = np.array(X).reshape(len(X), 1)
                else:
                    X = np.array(X[:, 0].reshape(len(X), 1))
                X = pd.DataFrame(X, columns=["smiles"])
                X.to_csv(data_path.name, index=False)
                with CaptureStdOut() as _:
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
        if self.num_workers > effective_n_jobs(-1):
            num_workers = effective_n_jobs(-1)
        else:
            num_workers = self.num_workers
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

        if fingerprint_type == "MPN":
            numpy_fp = np.zeros((len(X), self.hidden_size))
        elif fingerprint_type == "last_FFN":
            numpy_fp = np.zeros((len(X), self.ffn_hidden_size))
        else:
            raise ValueError("fingerprint_type should be one of ['MPN','last_FFN']")
        numpy_fp[:] = np.nan

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
                f"{num_workers}",
                "--fingerprint_type",
                f"{fingerprint_type}",
            ]
            # if self.x_aux_ is not None:
            # fprnt_args += ["--features_path", f"{x_aux_path.name}"] TODO: allow features once ChemProp is updated
            with CaptureStdOut() as _:
                args = chemprop.args.FingerprintArgs().parse_args(fprnt_args)
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
        sb = "ChemProp(" + ", ".join(sb) + ")"
        return sb


class ChemPropRegressor(RegressorMixin, BaseChemProp):
    def __init__(
        self,
        *,
        activation="ReLU",  # ReLU, LeakyReLU, PReLU, tanh, SELU, ELU
        aggregation="mean",  # mean, sum, norm
        aggregation_norm=100,
        aux_weight_pc=100,
        batch_size=50,
        dataset_type="regression",
        depth=3,
        dropout=0.0,
        ensemble_size=1,
        epochs=30,
        ffn_hidden_size=300,
        ffn_num_layers=3,
        final_lr_ratio_exp=-1,
        hidden_size=300,
        init_lr_ratio_exp=-1,
        max_lr_exp=-3,
        num_workers=-1,  # train pred num_workers>0 causes this https://github.com/facebookresearch/hydra/issues/964
        seed=0,
        side_info_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
        warmup_epochs_ratio=0.1,
        features_generator="none",  # none, morgan, morgan_count, rdkit_2d, rdkit_2d_normalized
    ):
        self.activation = activation
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm
        self.aux_weight_pc = aux_weight_pc
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.depth = depth
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn_num_layers = ffn_num_layers
        self.final_lr_ratio_exp = final_lr_ratio_exp
        self.hidden_size = hidden_size
        self.init_lr_ratio_exp = init_lr_ratio_exp
        self.max_lr_exp = max_lr_exp
        self.num_workers = effective_n_jobs(num_workers)
        self.warmup_epochs_ratio = warmup_epochs_ratio
        self.features_generator = features_generator
        self.seed = seed
        self.split_sizes = split_sizes
        self.side_info_rfe = side_info_rfe
        self.max_lr = 10**self.max_lr_exp
        self.init_lr = (10**self.init_lr_ratio_exp) * self.max_lr
        self.final_lr = (10**self.final_lr_ratio_exp) * self.max_lr


class ChemPropClassifier(ClassifierMixin, BaseChemProp):
    def __init__(
        self,
        *,
        activation="ReLU",  # ReLU, LeakyReLU, PReLU, tanh, SELU, ELU
        aggregation="mean",  # mean, sum, norm
        aggregation_norm=100,
        aux_weight_pc=100,
        batch_size=50,
        dataset_type="classification",
        depth=3,
        dropout=0.0,
        ensemble_size=1,
        epochs=30,
        ffn_hidden_size=300,
        ffn_num_layers=3,
        final_lr_ratio_exp=-1,
        hidden_size=300,
        init_lr_ratio_exp=-1,
        max_lr_exp=-3,
        num_workers=-1,  # train pred num_workers>0 causes this https://github.com/facebookresearch/hydra/issues/964
        seed=0,
        side_info_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
        warmup_epochs_ratio=0.1,
        features_generator="none",  # none, morgan, morgan_count, rdkit_2d, rdkit_2d_normalized
    ):
        self.activation = activation
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm
        self.aux_weight_pc = aux_weight_pc
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.depth = depth
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn_num_layers = ffn_num_layers
        self.final_lr_ratio_exp = final_lr_ratio_exp
        self.hidden_size = hidden_size
        self.init_lr_ratio_exp = init_lr_ratio_exp
        self.max_lr_exp = max_lr_exp
        self.num_workers = effective_n_jobs(num_workers)
        self.warmup_epochs_ratio = warmup_epochs_ratio
        self.features_generator = features_generator
        self.seed = seed
        self.split_sizes = split_sizes
        self.side_info_rfe = side_info_rfe
        self.max_lr = 10**self.max_lr_exp
        self.init_lr = (10**self.init_lr_ratio_exp) * self.max_lr
        self.final_lr = (10**self.final_lr_ratio_exp) * self.max_lr


class ChemPropPretrained(BaseChemProp):
    """
    Scikit-learn-like Chemprop for pretrained models
    """

    def __init__(self):
        pass

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
                raise NotImplementedError(
                    "side_info_rfe not supported for pretrained model"
                )
            else:
                si = process_side_info(self.side_info_)
            y = np.hstack((y, si))
        self.y_ = y
        self.target_columns = list(range(y.shape[1]))
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            extract_model_memory(self.model_, checkpoint_dir, self.pretrained_save_dir)
            with tempfile.TemporaryDirectory() as save_dir:
                self.save_dir = save_dir
                with tempfile.NamedTemporaryFile(delete=True, mode="w+") as data_path:
                    arguments = (
                        [
                            "--data_path",
                            f"{data_path.name}",
                            "--checkpoint_dir",
                            f"{checkpoint_dir}",
                            "--dataset_type",
                            self.dataset_type,
                            "--save_dir",
                            f"{save_dir}",
                            "--log_frequency",
                            "99999",
                            "--cache_cutoff",
                            "inf",
                            "--quiet",
                            "--seed",
                            self.seed,
                            "--pytorch_seed",
                            self.seed,
                            "--target_columns",
                        ]
                        + list(map(str, self.target_columns))
                        + [
                            "--activation",
                            self.activation,
                            "--aggregation",
                            self.aggregation,
                            "--aggregation_norm",
                            str(self.aggregation_norm),
                            "--batch_size",
                            str(self.batch_size),
                            "--depth",
                            str(int(self.depth)),
                            "--dropout",
                            str(int(self.dropout)),
                            "--ensemble_size",
                            str(int(self.ensemble_size)),
                            "--epochs",
                            str(int(self.epochs)),
                            "--ffn_hidden_size",
                            str(int(self.ffn_hidden_size)),
                            "--ffn_num_layers",
                            str(int(self.ffn_num_layers)),
                            "--final_lr",
                            str(self.final_lr),
                            "--hidden_size",
                            str(int(self.hidden_size)),
                            "--init_lr",
                            str(self.init_lr),
                            "--num_workers",
                            str(int(self.num_workers)),
                            "--max_lr",
                            str(self.max_lr),
                            "--split_sizes",
                        ]
                        + list(map(str, self.split_sizes))
                    )
                    if not torch.cuda.is_available():
                        arguments += ["--no_cuda"]
                    else:
                        arguments += ["--gpu", "0"]
                    if self.dataset_type == "classification":
                        arguments += ["--class_balance"]
                    weights_ = [
                        100 if targ == 0 else self.aux_weight_pc
                        for targ in self.target_columns
                    ]
                    self.target_weights = weights_
                    arguments += ["--target_weights"] + list(map(str, weights_))
                    if self.features_generator != "none":
                        arguments += [
                            "--features_generator",
                            f"{self.features_generator}",
                        ]
                        if self.features_generator == "rdkit_2d_normalized":
                            arguments += ["--no_features_scaling"]  # already pre-scaled
                    pd.DataFrame(
                        np.hstack((self.X_, self.y_)),
                        columns=["Smiles"] + list(map(str, range(self.y_.shape[1]))),
                    ).to_csv(data_path.name, index=False)
                    if self.x_aux_ is not None:
                        x_aux_path = tempfile.NamedTemporaryFile(
                            delete=True, mode="w+", suffix=".csv"
                        )
                        pd.DataFrame(
                            self.x_aux_,
                        ).to_csv(x_aux_path.name, index=False)
                    if self.frzn == "mpnn":
                        arguments += [
                            "--checkpoint_frzn",
                            checkpoint_dir + "/fold_0/model_0/model.pt",
                        ]
                    if self.frzn == "mpnn_first_ffn":
                        arguments += ["--frzn_ffn_layers", "1"]
                    if self.frzn == "mpnn_last_ffn":
                        arguments += ["--frzn_ffn_layers", f"{self.ffn_num_layers - 1}"]
                    with CaptureStdOut() as _:
                        args = QSARtunaTrainArgs().parse_args(arguments)
                        chemprop.train.cross_validate(
                            args=args, train_func=chemprop.train.run_training
                        )
                self.model_ = save_model_memory(save_dir)
            if self.x_aux_ is not None:
                x_aux_path.close()
        return self


class ChemPropRegressorPretrained(RegressorMixin, ChemPropPretrained):
    """
    Scikit-learn-like Chemprop for pretrained models
    """

    def __init__(
        self,
        *,
        dataset_type="regression",
        epochs=30,
        num_workers=-1,
        pretrained_model=None,
        frzn="none",  # ["none", "mpnn", "mpnn_first_ffn", "mpnn_last_ffn"]
    ):
        if pretrained_model is None:
            raise ValueError("Must provide a pretrained_model")
        if frzn not in ["none", "mpnn", "mpnn_first_ffn", "mpnn_last_ffn"]:
            raise ValueError(
                "freeze_checkpoint must be one of [None, 'mpnn', 'mpnn_first_ffn', 'mpnn_last_ffn']"
            )
        self.dataset_type = dataset_type
        self.epochs = epochs
        self.frzn = frzn
        self.num_workers = effective_n_jobs(num_workers)
        self.pretrained_model = pretrained_model

        with open(self.pretrained_model, "rb") as f:
            model = pickle.load(f)
            if not hasattr(model.predictor, "num_workers"):
                raise ValueError("Supplied model does not appear to be a ChemProp algo")
            for pre_param, pre_value in model.predictor.__dict__.items():
                if not hasattr(self, pre_param):
                    if pre_param == "save_dir":
                        self.pretrained_save_dir = pre_value
                    else:
                        self.__dict__[pre_param] = pre_value
                else:
                    if pre_param not in ["num_workers", "_estimator_type", "epochs"]:
                        init_value = self.__dict__[pre_param]
                        if pre_value != init_value:
                            raise ValueError(
                                f"pretrained {pre_param} is {pre_value} but {init_value} was supplied"
                            )
