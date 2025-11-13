import glob
import io
import logging
import math
import os
import tarfile
import tempfile
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import dill
import numpy as np
import pandas as pd
import rdkit.RDLogger as RDLogger
import torch
from chemprop import data
from chemprop.cli.fingerprint import FingerprintSubcommand
from chemprop.cli.predict import PredictSubcommand
from chemprop.cli.train import TrainSubcommand
from chemprop.featurizers.base import VectorFeaturizer
from chemprop.featurizers.molecule import MoleculeFeaturizerRegistry
from chemprop.models import MPNN
from configargparse import ArgumentParser, Namespace
from joblib import effective_n_jobs
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.Chem import Mol
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from optunaz.algorithms.side_info import binarise_side_info, process_side_info
from optunaz.descriptors import FeaturizerDescriptors, MordredDescriptors

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
rdlogger = RDLogger.logger()
rdlogger.setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")


@contextmanager
def suppress_logging(level=logging.CRITICAL):
    """Surpresses the output from ChemProp training/inference"""
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


@dataclass
class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

    This class is used to store information about a node in the MCTS tree,
    including the SMILES representation of the molecule, the set of atoms
    included in the subgraph, the total score (W), the number of visits (N),
    the prior probability (P), and the list of child nodes.
    It provides methods to calculate the quality (Q) of the node and the
    upper confidence bound (U) for exploration during the search process.

    The class expects: SMILES (input mols), atoms (set of atom indices in the subgraph),
    W (score accumulated from the node's visits), N (no. times a node is visited),
    P (prior probability of the node), and children (list child nodes in MCTS).
    """

    smiles: str
    atoms: Iterable[int]
    W: float = 0
    N: int = 0
    P: float = 0
    children: list[...] = field(default_factory=list)

    def __post_init__(self):
        self.atoms = set(self.atoms)

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int, c_puct: float = 10.0) -> float:
        return c_puct * self.P * math.sqrt(n) / (1 + self.N)


def mcts_make_prediction(
    models: list[MPNN],
    trainer: pl.Trainer,
    smiles: list[str],
) -> np.ndarray | None:
    """Generates predictions for a list of SMILES strings using a list of ChemProp models.

    This function takes a list of pretrained ChemProp models, a PyTorch Lightning trainer,
    and a list of SMILES strings as input. It processes the SMILES strings into a dataset,
    performs inference using the models, and returns the averaged predictions.
    """

    test_data = []
    idxs = []
    for smi_idx, smi in enumerate(smiles):
        try:
            test_data.append(data.MoleculeDatapoint.from_smi(smi))
            idxs.append(smi_idx)
        except RuntimeError:
            continue
    if test_data == []:
        return None
    test_dset = data.MoleculeDataset(test_data)
    test_loader = data.build_dataloader(
        test_dset, batch_size=1, num_workers=0, shuffle=False
    )

    with torch.inference_mode():
        sum_preds = []
        for model in models:
            predss = trainer.predict(model, test_loader)
            preds = torch.cat(predss, 0)
            preds = preds.cpu().numpy()
            sum_preds.append(preds)

        # Ensemble predictions
        sum_preds = sum(sum_preds)
        avg_preds = sum_preds / len(models)
    ret = np.zeros((len(idxs), preds.shape[-1]))
    ret[idxs] = avg_preds
    return ret


def mcts_find_clusters(
    mol: Chem.Mol,
) -> tuple[list[tuple[int, ...]], list[list[int]]]:
    """Identifies clusters of atoms and their associations in a molecule.

    Analyzes the input molecule to find clusters of atoms based on bonds
    and rings. It returns two outputs:
    1. A list of clusters, where each cluster is represented as a tuple of atom indices.
    2. A list of lists, where each sublist contains the indices of clusters associated
    with each atom.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def mcts_extract_subgraph_from_mol(
    mol: Chem.Mol, selected_atoms: set[int]
) -> tuple[Chem.Mol, list[int]]:
    """Extracts subgraph from a molecule based on selected atoms.

    RDKit molecule objects and a set of atom indices are used to extract
    a subgraph containing only the selected atoms. It adjusts aromaticity and removes
    atoms not in the selected set. Additionally, it identifies root atoms that connect
    the subgraph to the rest of the molecule.
    """
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [
            bond
            for bond in atom.GetBonds()
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
        ]
        aroma_bonds = [
            bond
            for bond in aroma_bonds
            if bond.GetBeginAtom().GetIdx() in selected_atoms
            and bond.GetEndAtom().GetIdx() in selected_atoms
        ]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [
        atom.GetIdx()
        for atom in new_mol.GetAtoms()
        if atom.GetIdx() not in selected_atoms
    ]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots


def mcts_extract_subgraph(
    smiles: str, selected_atoms: set[int]
) -> tuple[str, list[int]] | tuple[None, None]:
    """Extracts a subgraph from a SMILES based on selected atoms.

    SMILES string and a set of atom indices, and extracts
    a subgraph containing only the selected atoms. It attempts to kekulize the molecule
    and verifies the subgraph's validity. If the extraction fails, it retries without
    kekulization.
    """
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph, roots = mcts_extract_subgraph_from_mol(mol, selected_atoms)
    try:
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = Chem.MolFromSmiles(subgraph)
    except Exception:
        subgraph = None

    mol = Chem.MolFromSmiles(smiles)  # de-kekulize
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), roots

    # If fails, try without kekulization
    subgraph, roots = mcts_extract_subgraph_from_mol(mol, selected_atoms)
    subgraph = Chem.MolToSmiles(subgraph)
    subgraph = Chem.MolFromSmiles(subgraph)

    if subgraph is not None:
        return Chem.MolToSmiles(subgraph), roots
    else:
        return None, None


def mcts_rollout(
    node: MCTSNode,
    state_map: dict[str, MCTSNode],
    orig_smiles: str,
    clusters: list[tuple[int, ...]],
    atom_cls: list[list[int]],
    nei_cls: list[int],
    scoring_function: Callable[[str | list], list[float] | None],
    min_atoms: int,
    c_puct: float,
) -> float:
    """Performs a single rollout in the Monte Carlo Tree Search (MCTS) algorithm.

    Explores the search tree by selecting nodes based on their
    upper confidence bound (UCB) and expands unvisited nodes. It calculates
    scores for child nodes using the provided scoring function and updates
    the node's statistics.
    """
    cur_atoms = node.atoms
    if len(cur_atoms) <= min_atoms:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        # Cluster indices whose all atoms are present in current subgraph
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])

        for i in cur_cls:
            # Leaf atoms are atoms that are only involved in one cluster.
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]

            if (
                len(nei_cls[i] & cur_cls) == 1
                or len(clusters[i]) == 2
                and len(leaf_atoms) == 1
            ):
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = mcts_extract_subgraph(orig_smiles, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]  # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score

    sum_count = sum(c.N for c in node.children)
    selected_node = max(
        node.children, key=lambda x: x.Q() + x.U(sum_count, c_puct=c_puct)
    )
    v = mcts_rollout(
        selected_node,
        state_map,
        orig_smiles,
        clusters,
        atom_cls,
        nei_cls,
        scoring_function,
        min_atoms=min_atoms,
        c_puct=c_puct,
    )
    selected_node.W += v
    selected_node.N += 1

    return v


def mcts(
    smiles: str,
    scoring_function: Callable[[str | list], list | None],
    n_rollout: int,
    max_atoms: int,
    prop_delta: float,
    min_atoms: int,
    c_puct: float,
) -> list[MCTSNode]:
    """Performs Monte Carlo Tree Search (MCTS) to identify rationales in molecular structures.

    This function uses MCTS to explore subgraphs (rationales) of a molecule represented by a SMILES string.
    It evaluates the subgraphs using a scoring function and returns a list of nodes that meet the specified
    criteria.
    """
    mol = Chem.MolFromSmiles(smiles)

    clusters, atom_cls = mcts_find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    for _ in range(n_rollout):
        mcts_rollout(
            root,
            state_map,
            smiles,
            clusters,
            atom_cls,
            nei_cls,
            scoring_function,
            min_atoms=min_atoms,
            c_puct=c_puct,
        )

    rationales = [
        node
        for _, node in state_map.items()
        if len(node.atoms) <= max_atoms and node.P >= prop_delta
    ]

    return rationales


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


def register_featurizers():
    """
    Registers custom molecular featurizers for use with ChemProp.

    This function iterates through all `FeaturizerDescriptors` and dynamically creates
    and registers featurizer classes for each descriptor type. These featurizers are
    compatible with ChemProp and can process molecular data to generate features.

    The dynamically created featurizer classes include:
    - An `__init__` method to initialize the descriptor object.
    - A `__call__` method to calculate features from a molecule.
    - A `__len__` method to return the number of features.

    For MordredDescriptors, NaN features are replaced with a unique value (-999),
    and a binary mask is added to indicate valid features.

    The featurizers are registered with the `MoleculeFeaturizerRegistry` for use in ChemProp.
    """
    # Iterate through our own FeaturizerDescriptors and register them for ChemProp
    for descriptor_type in FeaturizerDescriptors.__args__:
        descriptor_object = descriptor_type.new()
        cls_name = f"{descriptor_type.__name__}Featurizer"

        def make_call(descriptor_type):
            def __call__(self, mol: Chem.Mol) -> np.ndarray:
                features = self.F.calculate_from_mol(mol)
                if isinstance(self.F, MordredDescriptors):
                    # For MordredDescriptors compatability, we assign NaN features a unique,
                    # obviously artificial value (-999) and an indicator binary column mask,
                    # marking whether the feature is valid
                    features[np.isnan(features)] = -999.0
                    valid_mask = ~np.isclose(features, -999.0)
                    features = np.hstack([features, valid_mask])
                return features

            return __call__

        def make_init(descriptor_object):
            def __init__(self):
                self.F = descriptor_object

            return __init__

        def __len__(self):
            return len(self.F)

        # Dynamically build the class dictionary
        cls_dict = {
            "__init__": make_init(descriptor_object),
            "__len__": __len__,
            "__call__": make_call(descriptor_type),
        }

        # Dynamically create each featurizer class
        DescriptorClass = types.new_class(
            cls_name,
            (VectorFeaturizer[Mol],),
            exec_body=lambda ns: ns.update(cls_dict),
        )

        # Register featurizer using registry instance
        MoleculeFeaturizerRegistry.register(descriptor_type.__name__.lower())(
            DescriptorClass
        )

register_featurizers()

def proccess_x(X):
    """
    Processes input data `X` to extract the main features, auxiliary labels, and auxiliary features.

    Input data, can be a single column or multiple columns. Multiple columns correspond to the following:
        1.) first column containes input SMILES (main feature)
        2.) second column (if present) is the Y-labels for task labels used in multi-task learning,
        3.) third column (if present) are the auxiliary descriptor features for molecules

    This function returns the processed SMILES (X), y_aux (additional tasks), X_aux (molecular descriptors)
    """

    X = np.array(X)
    X_aux = None
    y_aux = None

    if len(X.shape) == 1 or (X.shape[0] > 1 and X.shape[1] == 1):
        X = np.array(X).reshape(len(X), 1)
    else:
        try:
            y_aux = np.concatenate(X[:, 1])
        except (IndexError, ValueError):
            y_aux = X[:, 1]
        y_aux = None if (y_aux == None).all() else y_aux
        if X.shape[1] > 2:
            try:
                X_aux = np.concatenate(X[:, 2])
            except (IndexError, ValueError):
                X_aux = X[:, 2]
            X_aux = None if (X_aux == None).all() else X_aux
            if X_aux is not None:
                if (
                    X_aux.ndim == 1
                    or len(X_aux.shape) == 1
                    or (X_aux.shape[0] > 1 and X_aux.shape[1] == 1)
                ):
                    X_aux = X_aux.reshape(-1, 1)
        X = X[:, 0].reshape(-1, 1)

    return X, y_aux, X_aux


class BaseChemProp(BaseEstimator):
    """
    Base class for ChemProp models with Scikit-learn-like functionality.

    This class provides a unified interface for ChemProp models, including
    multitask learning capabilities and additional compound molecule features.

    It is inherited by `ChemPropRegressor`and `ChemPropClassifier` to reduce code duplication.

    The following attributes are possible:
        activation: Activation function used during trailing. Options include
            'RELU', 'LEAKYRELU', 'PRELU', 'TANH', 'SELU', 'ELU'.
        aggregation: Aggregation method for graph-level features. Options include
            'mean', 'sum', 'norm'.
        aggregation_norm: Normalization factor for aggregation.
        y_aux_weight_pc: Weight for auxiliary y labels (y_aux) in multitask learning (if applicable).
        batch_size: Number of samples per batch during training.
        batch_norm: Whether to use batch normalization.
        task_type: Type of task ('classification', 'regression' or 'regression-mve').
        depth: Number of message-passing layers
        dropout: Dropout rate
        ensemble_size: Number of ensembles.
        epochs: Maximum number epochs.
        patience: Number of epochs to wait before early stopping.
        ffn_hidden_dim: Hidden dimension size for feed-forward layers.
        ffn_num_layers: Number of feed-forward layers.
        final_lr: Final learning rate for training.
        final_lr_ratio: Ratio of final learning rate to maximum learning rate.
        message_hidden_dim: Hidden dimension size for message-passing layers.
        message_bias: Whether to use bias in message-passing layers.
        loss_function: Loss function to use during training.
        init_lr_ratio: Ratio of initial learning rate to maximum learning rate.
        init_lr: Initial learning rate for training.
        max_lr: Maximum learning rate for training.
        num_workers: Number of workers for data loading.
        seed: Random seed for reproducibility.
        corr_fe: Whether to use correlation-based feature elimination in multitask learning.
        y_aux_rfe: Whether to use recursive feature elimination (rfe) in multitask learning.
        split_sizes: Proportions for train, validation, and test splits.
        warmup_epochs_ratio: Ratio of warmup epochs to total epochs.
        molecule_featurizers: List of molecule featurizers to use.
        undirected: Whether to use undirected graphs in the model.
    """

    def __init__(
        self,
        activation="RELU",  # RELU, LEAKYRELU, PRELU, TANH, SELU, ELU
        aggregation="norm",  # mean, sum, norm
        aggregation_norm=100,
        y_aux_weight_pc=100,
        batch_size=64,
        batch_norm=False,
        task_type=None,
        depth=3,
        dropout=0.0,
        ensemble_size=1,
        epochs=100,
        patience=10,
        ffn_hidden_dim=300,
        ffn_num_layers=1,
        final_lr=0.0001,
        final_lr_ratio=1e-04,
        message_hidden_dim=300,
        message_bias=False,
        loss_function=None,
        init_lr_ratio=1e-04,
        init_lr=0.0001,
        max_lr=1e-03,
        num_workers=-1,
        seed=0,
        corr_fe=False,
        y_aux_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
        warmup_epochs_ratio=0.01,
        molecule_featurizers=None,  # morgan_binary, morgan_count, rdkit_2d, v1_rdkit_2d_normalized
        undirected=False,
    ):
        self.activation = activation
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm
        self.y_aux_weight_pc = y_aux_weight_pc
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.task_type = task_type
        self.depth = depth
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.patience = patience
        self.molecule_featurizers = molecule_featurizers
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_num_layers = ffn_num_layers
        self.final_lr = final_lr
        self.final_lr_ratio = final_lr_ratio
        self.message_hidden_dim = message_hidden_dim
        self.message_bias = message_bias
        self.loss_function = loss_function
        self.init_lr = init_lr
        self.init_lr_ratio = init_lr_ratio
        self.max_lr = max_lr
        self.num_workers = effective_n_jobs(num_workers)
        self.seed = seed
        self.split_sizes = split_sizes
        self.corr_fe = corr_fe
        self.y_aux_rfe = y_aux_rfe
        self.undirected = undirected
        self.warmup_epochs_ratio = warmup_epochs_ratio
        self.warmup_epochs = max(
            0, min(15, int(self.epochs * warmup_epochs_ratio))
        )  # to do: set 0 to 2
        if self.init_lr is None:
            self.init_lr = self.init_lr_ratio * self.max_lr
        else:
            # overwrite init_lr_ratio with reverse transform since init_lr takes priority
            # this allows base_chemprop_params to receive the base init_lr_ratio
            self.init_lr_ratio = self.init_lr / self.max_lr
        if self.final_lr is None:
            self.final_lr = self.final_lr_ratio * self.max_lr
        else:
            # overwrite final_lr_ratio with reverse transform since final_lr takes priority
            # this allows base_chemprop_params to receive the base init_lr_ratio
            self.final_lr_ratio = self.final_lr / self.max_lr

    def fit(self, X, y):
        """
        Processes the input features and target labels, prepares auxiliary data if available,
         and trains the model using the specified parameters.

         X are Input features, which may include SMILES strings, auxiliary labels,
         and auxiliary descriptor features.

         Y is simply the target labels for the primary task of intent.
        """
        self.X_ = X
        X, y_aux, X_aux = proccess_x(X)
        self.processed_X_ = X
        self.x_aux_ = X_aux
        self.descriptors_path_ = None

        if self.x_aux_ is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, mode="wb", suffix=".npz"
            ) as x_aux_path:
                np.savez(x_aux_path.name, self.x_aux_.astype(float))
                self.descriptors_path_ = x_aux_path.name

        y = np.array(y).reshape(-1, 1) if y.ndim == 1 else np.array(y)

        if is_classifier(self):
            self.classes_ = unique_labels(y).astype(np.uint8)
            if y_aux is not None:
                try:
                    _ = unique_labels(y_aux)
                except (ValueError, TypeError):
                    y_aux = process_side_info(
                        binarise_side_info(y_aux, cls=True),
                        y,
                        self.corr_fe,
                        self.y_aux_rfe,
                    )
            y = y.astype(np.uint8)
        else:
            if y_aux is not None:
                y_aux = binarise_side_info(y_aux)
                y_aux = process_side_info(
                    y_aux, y=y, corr_fs=self.corr_fe, rfe=self.y_aux_rfe
                )

        self.y_aux_ = y_aux
        if y_aux is not None:
            y = np.hstack((y, y_aux))
        self.y_ = y
        self.target_columns_ = list(map(str, range(y.shape[1])))
        self.task_weights_ = [
            100 if t == 0 else self.y_aux_weight_pc for t in self.target_columns_
        ]

        with tempfile.TemporaryDirectory() as output_dir:
            self.output_dir_ = output_dir

            with tempfile.NamedTemporaryFile(
                delete=True, mode="w+", suffix=".csv"
            ) as data_path:
                pd.DataFrame(
                    np.hstack((self.processed_X_, self.y_)),
                    columns=["Smiles"] + list(map(str, range(self.y_.shape[1]))),
                ).to_csv(data_path.name, index=False)

                args = Namespace(
                    data_path=Path(data_path.name),
                    output_dir=Path(output_dir),
                    task_type=self.task_type,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    patience=self.patience,
                    warmup_epochs=self.warmup_epochs,
                    num_workers=max(0, self.num_workers),
                    molecule_featurizers=self.molecule_featurizers
                    if self.molecule_featurizers is not None
                    else None,
                    no_descriptor_scaling=self.molecule_featurizers
                    != ["v1_rdkit_2d_normalized"],
                    dropout=self.dropout,
                    aggregation=self.aggregation,
                    aggregation_norm=self.aggregation_norm,
                    activation=self.activation,
                    init_lr=self.init_lr,
                    max_lr=self.max_lr,
                    final_lr=self.final_lr,
                    data_seed=self.seed,
                    pytorch_seed=self.seed,
                    ffn_hidden_dim=self.ffn_hidden_dim,
                    ffn_num_layers=self.ffn_num_layers,
                    depth=self.depth,
                    message_hidden_dim=self.message_hidden_dim,
                    message_bias=self.message_bias,
                    ensemble_size=self.ensemble_size,
                    split_sizes=list(self.split_sizes),
                    loss_function=self.loss_function,
                    target_columns=self.target_columns_,
                    class_balance=self.task_type == "classification",
                    split="RANDOM_WITH_REPEATED_SMILES",
                    no_batch_norm=not self.batch_norm,
                    task_weights=self.task_weights_,
                    undirected=self.undirected,
                    descriptors_path=self.descriptors_path_,
                    accelerator="auto",
                )

                parser = ArgumentParser()
                with suppress_logging():
                    parser = TrainSubcommand.add_args(
                        parser
                    )  # must set parser to result
                    self.train_args = parser.parse_args(args=[], namespace=args)
                    TrainSubcommand.func(self.train_args)

            self.model_ = save_model_memory(output_dir)

        if self.x_aux_ is not None:
            os.unlink(self.descriptors_path_)
        return self

    def predict_proba(self, X):
        """
        Predicts probabilities for the given input data.

        This method checks if the model has been fitted and then processes the input data
        to predict probabilities for classification tasks or regression values for regression tasks.

        X are input features, which may include SMILES strings, auxiliary labels,
        and auxiliary descriptor features.

        It returns  probabilities for classification tasks or predicted values for regression tasks.
        """
        check_is_fitted(self, ["model_"])
        X, y_aux, X_aux = proccess_x(X)
        descriptors_path = None
        num_workers = min(self.num_workers, effective_n_jobs(-1))

        if self.x_aux_ is not None:
            if X_aux is None:
                raise ValueError(
                    f"Model trained with auxiliary descriptors, please supply them with shape {self.x_aux_.shape}"
                )
            with tempfile.NamedTemporaryFile(
                delete=False, mode="wb", suffix=".npz"
            ) as x_aux_path:
                np.savez(x_aux_path.name, X_aux.astype(float))
                descriptors_path = x_aux_path.name

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_model_memory(self.model_, tmpdir, self.output_dir_)
            test_path = os.path.join(tmpdir, "test.csv")
            preds_path = os.path.join(tmpdir, "preds.pkl")
            pd.DataFrame(X, columns=["smiles"]).to_csv(test_path, index=False)

            parser = ArgumentParser()
            parser = PredictSubcommand.add_args(parser)
            arg_list = [
                "--test-path",
                test_path,
                "--model-path",
                tmpdir,
                "--output",
                preds_path,
                "--batch-size",
                str(self.batch_size),
                "--num-workers",
                str(num_workers),
            ]
            if self.molecule_featurizers:
                arg_list += ["--molecule-featurizers"] + self.molecule_featurizers
                if self.molecule_featurizers == ["v1_rdkit_2d_normalized"]:
                    arg_list.append("--no-features-scaling")
            if descriptors_path:
                arg_list += ["--descriptors-path", descriptors_path]

            with suppress_logging():
                PredictSubcommand.func(parser.parse_args(arg_list))

            preds = pd.read_pickle(preds_path).set_index("smiles")[["0"]].values

        if self.x_aux_ is not None:
            x_aux_path.close()
            os.unlink(x_aux_path.name)

        if is_classifier(self):
            yhats = np.zeros([len(preds), 2])
            yhats[:, 1] = preds.flatten()
            yhats[:, 0] = 1 - yhats[:, 1]
            return yhats
        else:
            return preds.flatten()

    def predict(self, X):
        """Predicts target values for the given input data.

        This method determines whether the model is a classifier or regressor and
        returns predictions accordingly. For classifiers, it returns binary predictions
        based on a threshold of 0.5. For regressors, it returns the predicted y-value &
        clips probabilistic predictions  to the range [0, 1] if target values are within range.

        Input is SMILES strings and (optionally) auxiliary labels, and auxiliary descriptor features.
        Outputs are class memberships or predicted regression values (clipped when probailistic), respectively.
        """
        if is_classifier(self):
            return self.predict_proba(X)[:, 1] > 0.5

        predictions = self.predict_proba(X).flatten()
        # clip probabilistic predictions
        return (
            predictions.clip(0, 1)
            if 0 <= self.y_.min() <= 1 and 0 <= self.y_.max() <= 1
            else predictions
        )

    def predict_uncert(self, X):
        """Predicts uncertainty for the given input data.

        Computes uncertainty estimates for predictions using one of the supported methods:
        'mve' for regression-mve tasks, 'ensemble' for ensemble models, or 'dropout' for single models.
        It processes the input data, validates auxiliary descriptors if applicable, and uses the ChemProp
        framework to generate predictions along with uncertainty values.

        Input features are SMILES strings and (optionally) auxiliary labels or auxiliary descriptor features.
        Outputs are predicted predictions followed by uncertainty estimates for the predictions.
        """
        check_is_fitted(self, ["model_"])
        X, y_aux, X_aux = proccess_x(X)
        num_workers = min(self.num_workers, effective_n_jobs(-1))

        if self.x_aux_ is not None:
            if X_aux is None:
                raise ValueError(
                    f"Model trained with auxiliary descriptors, please supply them with shape {self.x_aux_.shape}"
                )
            with tempfile.NamedTemporaryFile(
                delete=False, mode="wb", suffix=".npz"
            ) as x_aux_path:
                np.savez(x_aux_path.name, X_aux.astype(float))
                descriptors_path = x_aux_path.name

        uncertainty_method = (
            "mve"
            if self.task_type == "regression-mve"
            else ("ensemble" if self.ensemble_size > 1 else "dropout")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_model_memory(self.model_, tmpdir, self.output_dir_)
            test_path = os.path.join(tmpdir, "test.csv")
            preds_path = os.path.join(tmpdir, "preds.pkl")
            pd.DataFrame(X, columns=["smiles"]).to_csv(test_path, index=False)

            parser = ArgumentParser()
            parser = PredictSubcommand.add_args(parser)
            arg_list = [
                "--test-path",
                test_path,
                "--model-path",
                tmpdir,
                "--output",
                preds_path,
                "--batch-size",
                str(self.batch_size),
                "--num-workers",
                str(num_workers),
                "--uncertainty-method",
                uncertainty_method,
            ]
            if self.molecule_featurizers:
                arg_list += ["--molecule-featurizers"] + self.molecule_featurizers
                if self.molecule_featurizers == ["v1_rdkit_2d_normalized"]:
                    arg_list.append("--no-features-scaling")
            if self.x_aux_ is not None:
                arg_list += ["--descriptors-path", descriptors_path]

            with suppress_logging():
                PredictSubcommand.func(parser.parse_args(arg_list))

            preds = (
                pd.read_pickle(preds_path).set_index("smiles")[["0", "0_unc"]].values
            )

        return preds[:, 0], preds[:, 1]

    def interpret(
        self,
        X,
        prop_delta: float = 0.75,
        num_rationales_to_keep: int = 1,
        rollout: int = 10,
        c_puct: float = 10.0,
        max_atoms: int = 20,
        min_atoms: int = 8,
        property_id: int = 0,
    ):
        """
        Performs model interpretation by identifying rationales for predictions.

        This method uses Monte Carlo Tree Search (MCTS) to extract subgraphs (rationales)
        from molecular structures that contribute significantly to the predicted property.
        It generates rationales based on the specified parameters and returns a DataFrame
        containing the original SMILES strings, predictions, and rationales.

        Inputs are SMILES strings representing molecular structures (not supported for Y/X auxiliary data).
        The input variables are:
            prop_delta (minimum property score threshold for rationales).
            num_rationales_to_keep (top rationales to retain for each molecule)
            rollout (number MCTS rollouts to perform)
            c_puct (exploration constant for MCTS)
            max_atoms (maximum number of atoms allowed in a rationale)
            min_atoms (minimum number of atoms required in a rationale)
            property_id (Index of the property to interpret)

        The output is a dataFrame containing the SMILES , predictions, and rationales with scores.
        """

        check_is_fitted(self, ["model_"])

        if self.x_aux_ is not None:
            raise ValueError(
                f"Model trained with auxiliary descriptors, which is not currently supported for interpret"
            )

        X, y_aux, X_aux = proccess_x(X)

        if y_aux is not None:
            raise ValueError(
                f"Side information y labels provided but not currently supported for interpret"
            )

        if X_aux is not None:
            raise ValueError(
                f"Auxiliary x descriptors provided but not currently supported for interpret"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_model_memory(self.model_, tmpdir, self.output_dir_)
            mpnn = MPNN.load_from_file(f"{tmpdir}/model_0/best.pt", map_location="cpu")
            trainer = pl.Trainer(
                logger=False, enable_progress_bar=False, accelerator="cpu"
            )
            models = [mpnn] if type(mpnn) == MPNN else mpnn
            results_df = {"smiles": [], "prediction": []}

            for i in range(num_rationales_to_keep):
                results_df[f"rationale_{i}"] = []
                results_df[f"rationale_{i}_score"] = []

            for smiles in X.flatten():

                def scoring_function(smiles: str | list) -> list | None:
                    pred = mcts_make_prediction(
                        models=models,
                        trainer=trainer,
                        smiles=[smiles] if type(smiles) in [str, np.str_] else smiles,
                    )
                    return pred if pred is None else pred[:, property_id]

                score = scoring_function(smiles)[0]
                if (score is not None) and (score > prop_delta):
                    rationales = mcts(
                        smiles=smiles,
                        scoring_function=scoring_function,
                        n_rollout=rollout,
                        max_atoms=max_atoms,
                        prop_delta=prop_delta,
                        min_atoms=min_atoms,
                        c_puct=c_puct,
                    )
                else:
                    rationales = []

                results_df["smiles"].append(smiles)
                results_df["prediction"].append(score)

                if len(rationales) == 0:
                    for i in range(num_rationales_to_keep):
                        results_df[f"rationale_{i}"].append(None)
                        results_df[f"rationale_{i}_score"].append(None)
                else:
                    min_size = min(len(x.atoms) for x in rationales)
                    min_rationales = [x for x in rationales if len(x.atoms) == min_size]
                    rats = sorted(min_rationales, key=lambda x: x.P, reverse=True)

                    for i in range(num_rationales_to_keep):
                        if i < len(rats):
                            results_df[f"rationale_{i}"].append(rats[i].smiles)
                            results_df[f"rationale_{i}_score"].append(rats[i].P)
                        else:
                            results_df[f"rationale_{i}"].append(None)
                            results_df[f"rationale_{i}_score"].append(None)
        return pd.DataFrame(results_df)

    def chemprop_fingerprint(self, X, fingerprint_type="MPN"):
        """Generates molecular fingerprints using the ChemProp framework.

        This method computes fingerprints for molecular structures based on the specified
        fingerprint type ('MPN' or 'last_FFN'). It processes the input data, validates auxiliary
        descriptors if applicable, and uses the ChemProp framework to generate fingerprints.

        Input is SMILES strings and (optionally) auxiliary labels or auxiliary descriptor features.
        The fingerprint_type defines the fingerprint to generate, and can be 'MPN' (message-passing
        network fingerprints), 'last_FFN' (fingerprints from the last feed-forward layer)

        The output are generated molecular fingerprints, with the shape (len(X), message_hidden_dim
        + [any molecule additional descriptors dim]) for MPN and (len(X), ffn_hidden_dim) for last_FFN.
        """

        check_is_fitted(self, ["model_"])
        X, y_aux, X_aux = proccess_x(X)

        if self.x_aux_ is not None:
            if X_aux is None:
                raise ValueError(
                    f"Model trained with auxiliary descriptors, please supply them with shape {self.x_aux_.shape}"
                )
            with tempfile.NamedTemporaryFile(
                delete=False, mode="wb", suffix=".npz"
            ) as x_aux_path:
                # Save auxiliary descriptors as a numpy array
                np.savez(x_aux_path.name, X_aux.astype(float))
                descriptors_path = x_aux_path.name

        num_workers = min(self.num_workers, effective_n_jobs(-1))
        ffn_block_index = {"MPN": "0", "last_FFN": "-1"}[fingerprint_type]

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_model_memory(self.model_, tmpdir, self.output_dir_)
            test_path = os.path.join(tmpdir, "test.csv")
            fps_path = os.path.join(tmpdir, "fps.npz")
            pd.DataFrame(X, columns=["smiles"]).to_csv(test_path, index=False)

            parser = ArgumentParser()
            parser = FingerprintSubcommand.add_args(parser)
            arg_list = [
                "--test-path",
                test_path,
                "--model-path",
                tmpdir,
                "--output",
                fps_path,
                "--batch-size",
                str(self.batch_size),
                "--num-workers",
                str(self.num_workers),
                "--ffn-block-index",
                ffn_block_index,
            ]
            if self.molecule_featurizers:
                arg_list += ["--molecule-featurizers"] + self.molecule_featurizers
                if self.molecule_featurizers == ["v1_rdkit_2d_normalized"]:
                    arg_list.append("--no-features-scaling")
            if self.x_aux_ is not None:
                arg_list += ["--descriptors-path", descriptors_path]

            with suppress_logging():
                FingerprintSubcommand.func(parser.parse_args(arg_list))

            fps_path = Path(fps_path).parent / (Path(fps_path).stem + "_0.npz")
            fps = np.load(fps_path)["H"]

        return fps

    def __str__(self):
        do_not_print = {
            "X_",
            "y_",
            "x_aux_",
            "num_workers",
            "hash_",
            "model_",
            "train_args",
            "y_aux_",
            "target_columns",
            "target_weight",
        }
        attributes = [
            f"{key}='{value}'"
            for key, value in self.__dict__.items()
            if key not in do_not_print
        ]
        return f"ChemProp({', '.join(attributes)})"


class ChemPropRegressor(RegressorMixin, BaseChemProp):
    def __init__(
        self,
        activation="RELU",
        aggregation="norm",
        aggregation_norm=100,
        y_aux_weight_pc=100,
        batch_size=64,
        batch_norm=False,
        task_type=None,
        depth=3,
        dropout=0.0,
        ensemble_size=1,
        epochs=100,
        patience=10,
        ffn_hidden_dim=300,
        ffn_num_layers=1,
        final_lr=None,
        final_lr_ratio=1e-04,
        message_hidden_dim=300,
        message_bias=False,
        loss_function="mse",
        init_lr=None,
        init_lr_ratio=1e-04,
        max_lr=1e-03,
        num_workers=-1,
        seed=0,
        corr_fe=False,
        y_aux_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
        warmup_epochs_ratio=0.01,
        molecule_featurizers=None,
        undirected=False,
    ):
        self.activation = activation
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm
        self.y_aux_weight_pc = y_aux_weight_pc
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.task_type = (
            task_type
            if task_type is not None
            else "regression-mve"
            if loss_function == "mve"
            else "regression"
        )
        self.depth = depth
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.patience = patience
        self.molecule_featurizers = molecule_featurizers
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_num_layers = ffn_num_layers
        self.final_lr = final_lr
        self.final_lr_ratio = final_lr_ratio
        self.message_hidden_dim = message_hidden_dim
        self.message_bias = message_bias
        self.loss_function = loss_function
        self.init_lr = init_lr
        self.init_lr_ratio = init_lr_ratio
        self.max_lr = max_lr
        self.num_workers = effective_n_jobs(num_workers)
        self.seed = seed
        self.split_sizes = split_sizes
        self.corr_fe = corr_fe
        self.y_aux_rfe = y_aux_rfe
        self.undirected = undirected
        self.warmup_epochs_ratio = warmup_epochs_ratio
        self.warmup_epochs = max(
            0, min(15, int(self.epochs * warmup_epochs_ratio))
        )  # to do: set 0 to 2
        if self.init_lr is None:
            self.init_lr = self.init_lr_ratio * self.max_lr
        if self.final_lr is None:
            self.final_lr = self.final_lr_ratio * self.max_lr


class ChemPropClassifier(ClassifierMixin, BaseChemProp):
    def __init__(
        self,
        activation="RELU",
        aggregation="norm",
        aggregation_norm=100,
        y_aux_weight_pc=100,
        batch_size=64,
        batch_norm=False,
        task_type="classification",
        depth=3,
        dropout=0.0,
        ensemble_size=1,
        epochs=100,
        patience=10,
        ffn_hidden_dim=300,
        ffn_num_layers=1,
        final_lr=None,
        final_lr_ratio=1e-04,
        message_hidden_dim=300,
        message_bias=False,
        loss_function="bce",
        init_lr=None,
        init_lr_ratio=1e-04,
        max_lr=1e-03,
        num_workers=-1,
        seed=0,
        corr_fe=False,
        y_aux_rfe=False,
        split_sizes=(0.8, 0.2, 0.0),
        warmup_epochs_ratio=0.01,
        molecule_featurizers=None,
        undirected=False,
    ):
        self.activation = activation
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm
        self.y_aux_weight_pc = y_aux_weight_pc
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.task_type = task_type
        self.depth = depth
        self.dropout = dropout
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.patience = patience
        self.molecule_featurizers = molecule_featurizers
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_num_layers = ffn_num_layers
        self.final_lr = final_lr
        self.final_lr_ratio = final_lr_ratio
        self.message_hidden_dim = message_hidden_dim
        self.message_bias = message_bias
        self.loss_function = loss_function
        self.init_lr = init_lr
        self.init_lr_ratio = init_lr_ratio
        self.max_lr = max_lr
        self.num_workers = effective_n_jobs(num_workers)
        self.seed = seed
        self.split_sizes = split_sizes
        self.corr_fe = corr_fe
        self.y_aux_rfe = y_aux_rfe
        self.undirected = undirected
        self.warmup_epochs_ratio = warmup_epochs_ratio
        self.warmup_epochs = max(
            0, min(15, int(self.epochs * warmup_epochs_ratio))
        )  # to do: set 0 to 2
        if self.init_lr is None:
            self.init_lr = self.init_lr_ratio * self.max_lr
        if self.final_lr is None:
            self.final_lr = self.final_lr_ratio * self.max_lr


class ChemPropPretrained(BaseChemProp):
    """Scikit-learn-like Chemprop for pretrained models

    This module provides functionality for using pretrained ChemProp models
    in a Scikit-learn-like interface. It supports both regression and classification tasks
    and allows for fine-tuning or freezing specific parts of the pretrained model.

    Key features:
        - Supports freezing specific layers or components of the pretrained model.
        - Allows for additional training with new data.
        - Provides compatibility with Scikit-learn estimators.

    The following are used at initiation:
        - model_path: Path to the pretrained ChemProp model directory.
        - batch_size: Batch size for training and prediction.
        - num_workers: Number of workers for data loading.
        - task_type: Type of task ('classification', 'regression' or 'regression-mve').
        - epochs: Number of epochs for training (0 means no training).
        - patience: Number of epochs with no improvement after which training will be stopped.
        - frzn_layers: List of layers to freeze during training
    The following inputs are expected are train time:
        - X: New SMILES strings and the auxiliary labels for X and Y to transfer learn to
        - y: New target values to transfer learn to

    When performing transfer learning, the model will be updated with the new data, and the frzn_layers
    dictate which parts of the model will remain unchanged during this process.
    The following options are available:
        - None: No layers are frozen, all layers are trainable.
        - "mpnn": MPNN layers frozen - only the feed-forward network is trained.
        - "first_ffn": First feed-forward layer frozen - all subsequent layers trained.
        - "last_ffn": Last feed-forward layer frozen - all other layers to be trained.
        - "mpnn_first_ffn": Freezes MPNN layers and the first feed-forward layer - all other layers are trained.
        - "mpnn_last_ffn": Freezes MPNN layers and the last feed-forward layer - all other layers trained.
    """

    def fit(self, X, y):
        if self.epochs == 0:
            # If epochs is 0, we do not refit the model, or update any attributes
            return self
        try:
            self.X_ = np.concatenate((self.X_, X))
        except ValueError:
            raise ValueError(
                f"Model trained with X shape {self.X_.shape}, but got {X.shape}"
            )
        X, y_aux, X_aux = proccess_x(X)

        # Validate auxiliary descriptors
        if self.x_aux_ is not None and X_aux is None:
            raise ValueError(
                f"Model trained with auxiliary descriptors, please supply them with shape {self.x_aux_.shape}"
            )

        # Validate auxiliary descriptors
        if self.y_aux_ is not None:
            if y_aux is None:
                raise ValueError(
                    f"Model trained with auxiliary labels, please supply them with shape {self.y_aux_.shape}"
                )
            if self.y_aux_rfe or self.corr_fe:
                raise NotImplementedError(
                    "y_aux_rfe and corr_fe not supported for pretrained model"
                )

        self.processed_X_ = np.concatenate((self.processed_X_, X))

        # Save auxiliary descriptors to a temporary file
        self.descriptors_path_ = None
        if X_aux is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, mode="wb", suffix=".npz"
            ) as x_aux_path:
                np.savez(x_aux_path.name, X_aux.astype(float))
                self.descriptors_path_ = x_aux_path.name
            self.x_aux_ = np.concatenate((self.x_aux_, X_aux))

        y = np.array(y).reshape(-1, 1) if y.ndim == 1 else np.array(y)

        # Handle classification or regression tasks
        if is_classifier(self):
            self.classes_ = unique_labels(y).astype(np.uint8)
            if y_aux is not None:
                try:
                    _ = unique_labels(y_aux)
                except (ValueError, TypeError):
                    y_aux = process_side_info(
                        binarise_side_info(y_aux, cls=True),
                        y,
                        self.corr_fe,
                        self.y_aux_rfe,
                    )
            y = y.astype(np.uint8)
        else:
            if y_aux is not None:
                y_aux = binarise_side_info(y_aux)
                y_aux = process_side_info(
                    y_aux, y=y, corr_fs=self.corr_fe, rfe=self.y_aux_rfe
                )
                self.y_aux_ = np.concatenate((self.y_aux_, y_aux))

        # Combine target and side information
        if y_aux is not None:
            y = np.hstack((y, y_aux))

        try:
            self.y_ = np.concatenate((self.y_, y))
        except ValueError:
            raise ValueError(
                f"Expected y columns ({y.shape[1]}) to be same as y_ ({self.y_.shape[1]})"
            )

        self.target_columns_ = list(map(str, range(y.shape[1])))
        self.task_weights_ = [
            100 if t == 0 else self.y_aux_weight_pc for t in self.target_columns_
        ]

        with tempfile.TemporaryDirectory() as output_dir:
            extract_model_memory(self.model_, output_dir, self.pretrained_output_dir_)
            self.output_dir_ = output_dir
            with tempfile.NamedTemporaryFile(
                delete=True, mode="w+", suffix=".csv"
            ) as data_path:
                pd.DataFrame(
                    np.hstack((X, y)),
                    columns=["Smiles"] + list(map(str, range(self.y_.shape[1]))),
                ).to_csv(data_path.name, index=False)

                args = Namespace(
                    data_path=Path(data_path.name),
                    output_dir=Path(output_dir),
                    checkpoint=[
                        Path(i) for i in glob.glob(f"{output_dir}/model_*/best.pt")
                    ],
                    task_type=self.task_type,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    patience=self.patience,
                    warmup_epochs=self.warmup_epochs,
                    num_workers=max(0, self.num_workers),
                    molecule_featurizers=self.molecule_featurizers,
                    no_descriptor_scaling=self.molecule_featurizers
                    != ["v1_rdkit_2d_normalized"],
                    dropout=self.dropout,
                    aggregation=self.aggregation,
                    aggregation_norm=self.aggregation_norm,
                    activation=self.activation,
                    init_lr_ratio=self.init_lr_ratio,
                    max_lr=self.max_lr,
                    final_lr_ratio=self.final_lr_ratio,
                    data_seed=self.seed,
                    pytorch_seed=self.seed,
                    ffn_hidden_dim=self.ffn_hidden_dim,
                    ffn_num_layers=self.ffn_num_layers,
                    depth=self.depth,
                    message_hidden_dim=self.message_hidden_dim,
                    message_bias=self.message_bias,
                    ensemble_size=self.ensemble_size,
                    split_sizes=list(self.split_sizes),
                    loss_function=self.loss_function,
                    target_columns=self.target_columns_,
                    class_balance=self.task_type == "classification",
                    split="RANDOM_WITH_REPEATED_SMILES",
                    no_batch_norm=not self.batch_norm,
                    task_weights=self.task_weights_,
                    undirected=self.undirected,
                    descriptors_path=self.descriptors_path_,
                    accelerator="auto",
                    freeze_encoder=0 if self.frzn is None else ("mpnn" in self.frzn),
                    frzn_ffn_layers=0
                    if self.frzn is None
                    else 1
                    if "first" in self.frzn
                    else -1,
                )

                parser = ArgumentParser()
                parser = TrainSubcommand.add_args(parser)
                with suppress_logging():
                    self.train_args = parser.parse_args(args=[], namespace=args)
                    TrainSubcommand.func(self.train_args)

            self.model_ = save_model_memory(output_dir)

        # Clean up temporary files
        if self.x_aux_ is not None:
            os.unlink(self.descriptors_path_)

        return self


class ChemPropRegressorPretrained(RegressorMixin, ChemPropPretrained):
    """
    Scikit-learn-like Chemprop for pretrained models
    """

    def __init__(
        self,
        *,
        task_type="regression-mve",
        epochs=100,
        patience=10,
        num_workers=-1,
        pretrained_model=None,
        frzn=None,
    ):
        if pretrained_model is None:
            raise ValueError("Must provide a pretrained_model")
        if frzn not in [
            None,
            "mpnn",
            "mpnn_first_ffn",
            "mpnn_last_ffn",
            "first_ffn",
            "last_ffn",
        ]:
            raise ValueError(
                "freeze_checkpoint must be one of [None, 'mpnn', 'mpnn_first_ffn', 'mpnn_last_ffn']"
            )
        self.task_type = task_type
        self.epochs = epochs
        self.patience = patience
        self.frzn = frzn
        self.num_workers = effective_n_jobs(num_workers)
        self.pretrained_model = pretrained_model

        with open(self.pretrained_model, "rb") as f:
            model = dill.load(f)
            if not hasattr(model.predictor, "num_workers"):
                raise ValueError("Supplied model does not appear to be a ChemProp algo")
            for pre_param, pre_value in model.predictor.__dict__.items():
                if not hasattr(self, pre_param):
                    if pre_param == "output_dir_":
                        self.pretrained_output_dir_ = pre_value
                        # also set output_dir to the same value until we refit (catches epochs=0 not running fit)
                        self.output_dir_ = pre_value
                    else:
                        self.__dict__[pre_param] = pre_value
                else:
                    if pre_param not in [
                        "num_workers",
                        "epochs",
                        "patience",
                        "frzn",
                        "pretrained_model",
                    ]:
                        init_value = self.__dict__[pre_param]
                        if pre_value != init_value:
                            if pre_param == "task_type" and pre_value in [
                                "regression",
                                "regression-mve",
                            ]:
                                self.__dict__[pre_param] = pre_value
                            else:
                                raise ValueError(
                                    f"pretrained {pre_param} is {pre_value} but {init_value} was supplied"
                                )


class ChemPropClassifierPretrained(ClassifierMixin, ChemPropPretrained):
    """
    Scikit-learn-like Chemprop for pretrained models
    """

    def __init__(
        self,
        *,
        task_type="classification",
        epochs=100,
        patience=10,
        num_workers=-1,
        pretrained_model=None,
        frzn=None,
    ):
        if pretrained_model is None:
            raise ValueError("Must provide a pretrained_model")
        if frzn not in [
            None,
            "mpnn",
            "mpnn_first_ffn",
            "mpnn_last_ffn",
            "first_ffn",
            "last_ffn",
        ]:
            raise ValueError(
                "freeze_checkpoint must be one of [None, 'mpnn', 'mpnn_first_ffn', 'mpnn_last_ffn']"
            )
        self.task_type = task_type
        self.epochs = epochs
        self.patience = patience
        self.frzn = frzn
        self.num_workers = effective_n_jobs(num_workers)
        self.pretrained_model = pretrained_model

        with open(self.pretrained_model, "rb") as f:
            model = dill.load(f)
            if not hasattr(model.predictor, "num_workers"):
                raise ValueError("Supplied model does not appear to be a ChemProp algo")
            for pre_param, pre_value in model.predictor.__dict__.items():
                if not hasattr(self, pre_param):
                    if pre_param == "output_dir_":
                        self.pretrained_output_dir_ = pre_value
                        # also set output_dir to the same value until we refit (catches epochs=0 not running fit)
                        self.pre_param = pre_value
                    else:
                        self.__dict__[pre_param] = pre_value
                else:
                    if pre_param not in [
                        "num_workers",
                        "epochs",
                        "patience",
                        "frzn",
                        "pretrained_model",
                    ]:
                        init_value = self.__dict__[pre_param]
                        if pre_value != init_value:
                            raise ValueError(
                                f"pretrained {pre_param} is {pre_value} but {init_value} was supplied"
                            )
