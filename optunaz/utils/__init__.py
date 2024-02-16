import dataclasses
import json
from typing import Any, Dict
import numpy as np
from pathlib import Path
import pandas as pd
import hashlib

def mkdict(obj: Any) -> Dict:
    # To recursively convert nested dataclasses to dict, use json machinery.

    # https://stackoverflow.com/a/51286749
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    objstr = json.dumps(obj, cls=EnhancedJSONEncoder)
    objdict = json.loads(objstr)
    return objdict


def load_df_from_file(filename: str, smiles_col: str):
    file_format = Path(filename).suffix
    if file_format == ".csv":
        try:
            return pd.read_csv(filename, skipinitialspace=True, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(
                filename, skipinitialspace=True, low_memory=False, encoding="latin"
            )
    elif file_format == ".sdf":
        # Import here so as not to "spill" dependencies into pickled files
        from rdkit.Chem import PandasTools

        return PandasTools.LoadSDF(filename, smilesName=smiles_col, isomericSmiles=True)
    else:
        raise ValueError(f"Unsupported format for data: {file_format}.")


def remove_failed_idx(failed_idx, y_, smis, auxs) -> tuple[Any, Any, Any]:
    y_ = np.array([val for y_idx, val in enumerate(y_) if y_idx not in failed_idx])
    smis = np.array([smi for s_idx, smi in enumerate(smis) if s_idx not in failed_idx])
    if auxs is not None:
        auxs = np.array(
            [aux for s_idx, aux in enumerate(auxs) if s_idx not in failed_idx],
            dtype=float,
        )
        if len(auxs.shape) == 1:
            auxs = auxs.reshape(len(auxs), 1)
    else:
        auxs = None
    return y_, smis, auxs

def md5_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of an optimisation algorithm or a model metadata dictionary"""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()