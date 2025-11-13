import copy
import json

import numpy as np
import pandas as pd
import shap
from apischema import serialize
from joblib import Parallel, delayed, effective_n_jobs
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import is_classifier

from optunaz.config import ModelMode
from optunaz.descriptors import ECFP, ECFP_counts, PathFP
from optunaz.utils.preprocessing.splitter import StratifiedOverlappingGroupKFold


def get_ecfp_fpinfo(m, descriptor):
    """Return the ecfp info for a compound mol"""

    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateAtomCounts()
    ao.AllocateAtomToBits()
    ao.AllocateBitInfoMap()
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=descriptor.parameters.radius,
        fpSize=descriptor.parameters.nBits,
    )
    mfpgen.GetCountFingerprint(m, additionalOutput=ao) if isinstance(
        descriptor, ECFP_counts
    ) else mfpgen.GetFingerprint(m, additionalOutput=ao)
    return ao.GetBitInfoMap()


def get_pathfp_fpinfo(m, descriptor):
    """Return the ecfp_count info for a compound mol"""

    fp = rdFingerprintGenerator.GetRDKitFPGenerator(
        maxPath=descriptor.parameters.maxPath, fpSize=descriptor.parameters.fpSize
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.CollectBitPaths()
    fp.GetFingerprint(m, additionalOutput=ao)
    return ao.GetBitPaths()


def explain_ECFP_PathFP(len_feats, estimator, descriptor):
    """Explain ECFPs & PathFP using train atom environments"""
    # Initialize an empty array to store the results
    ret = np.empty(len_feats, dtype="<U50")

    # Iterate through each important feature index
    for feat_idx in range(len_feats):
        feature_explained = False  # Flag to check if the feature is explained
        # Iterate through each molecule in the training set for compounds with that feature
        for mol_idx, mol in enumerate(estimator.X_):
            # Check if the feature is present in the molecule
            if mol[feat_idx] > 0:
                m = Chem.MolFromSmiles(estimator.train_smiles_[mol_idx])
                # Get the feature information based on the descriptor type
                info = (
                    get_ecfp_fpinfo(m, descriptor).get(feat_idx, [])
                    if isinstance(descriptor, (ECFP, ECFP_counts))
                    else get_pathfp_fpinfo(m, descriptor).get(feat_idx, [])
                )
                # Iterate through the feature information, breaking when valid smiles produced
                for env in info:
                    try:
                        # Handle PathFP descriptor
                        if isinstance(descriptor, PathFP):
                            atoms = {m.GetBondWithIdx(e).GetBeginAtomIdx() for e in env}
                            atoms.update(
                                m.GetBondWithIdx(e).GetEndAtomIdx() for e in env
                            )
                            feat_smi = Chem.MolFragmentToSmiles(
                                m, atomsToUse=list(atoms), bondsToUse=env
                            )
                        # Handle ECFP and ECFP_counts descriptors
                        else:
                            atom, radius = env
                            eofradius = Chem.FindAtomEnvironmentOfRadiusN(
                                m, radius, atom
                            )
                            amap = {}
                            submol = Chem.PathToSubmol(m, eofradius, atomMap=amap)
                            feat_smi = Chem.MolToSmiles(
                                submol, rootedAtAtom=amap[atom], canonical=False
                            )
                        # Store the feature SMILES if it is valid
                        if feat_smi:
                            ret[feat_idx] = feat_smi
                            feature_explained = True  # Set the flag to True
                            break
                    # Handle exceptions when MolToSmiles or MolFragmentToSmiles fails
                    except (KeyError, RuntimeError):
                        pass
            if feature_explained:
                break  # Break out of the inner loop if the feature is explained
    return ret


def get_fp_info(exp_df, estimator, descript, fp_idx, strt_idx=None):
    """Get details for the descriptors when available"""
    info = (
        explain_ECFP_PathFP(fp_idx, estimator, descript)
        if isinstance(descript, (ECFP, ECFP_counts, PathFP))
        else descript.parameters.feature_names
        if hasattr(descript.parameters, "feature_names")
        else descript.parameters.descriptor.parameters.feature_names
        if hasattr(descript.parameters, "descriptor")
        and hasattr(descript.parameters.descriptor.parameters, "feature_names")
        else []
    )

    if len(info) > 0:
        if strt_idx is not None:
            exp_df.loc[strt_idx : strt_idx + fp_idx - 1, "info"] = info
        else:
            exp_df["info"] = info
    return


def runPermutation(X_pred, mode, estimator):
    max_evals = 2 * X_pred.shape[1] + 1
    if mode == ModelMode.REGRESSION:
        explainer = shap.explainers.Permutation(
            estimator.predict, estimator.X_, max_evals=max_evals, silent=True
        )
    if mode == ModelMode.CLASSIFICATION:
        explainer = shap.explainers.Permutation(
            estimator.predict_proba, estimator.X_, max_evals=max_evals, silent=True
        )
    return np.abs(np.array(explainer(X_pred, silent=True).values))


def runShap(estimator, X_pred, mode):
    """Explain model prediction using auto explainer or SHAP KernelExplainer"""
    X_pred = X_pred.astype(np.float64)

    # Estimators employing feature selection handled here to minimize the SHAP latency
    if hasattr(estimator, "_important_feats"):
        X_pred = X_pred[:, estimator._important_feats]
        estimator.X_ = estimator.X_[:, estimator._important_feats]
        estimator.n_features_in_ = len(estimator._important_feats)
        del estimator._important_feats
        return runPermutation(X_pred, mode, estimator)

    try:
        # Attempt to use SHAP's auto explainer
        explainer = shap.Explainer(estimator, estimator.X_)
        return np.abs(np.array(explainer(X_pred).values))
    except TypeError:
        try:
            # Handle models requiring specific inference methods
            explainer = shap.Explainer(
                estimator.predict
                if mode == ModelMode.REGRESSION
                else estimator.predict_proba,
                estimator.X_,
            )
            return np.abs(
                np.array(
                    explainer(
                        X_pred, silent=True, max_evals=2 * X_pred.shape[1] + 1
                    ).values
                )
            )
        except ValueError:
            # Fallback explicitly to permutation explainer
            try:
                return runPermutation(X_pred, mode, estimator)
            except AttributeError:
                # Use KernelExplainer for unsupported models
                explainer = shap.KernelExplainer(
                    estimator.predict
                    if mode == ModelMode.REGRESSION
                    else estimator.predict_proba,
                    estimator.X_,
                    silent=True,
                )
                return np.abs(np.array(explainer(X_pred, nsamples="auto").values))


def ShapExplainer(estimator, X_pred, mode, descriptor):
    """
    Run SHAP and populate the explainability dataframe
    """
    important_feats = getattr(estimator, "_important_feats", None)

    # Create copy of estimator and descriptor since we modify them within other functions
    estimator_ = copy.deepcopy(estimator)
    descriptor_ = copy.deepcopy(descriptor)

    shap_values = runShap(estimator_, X_pred, mode)

    # process the shap_values shapes
    # if explainer explains both classes, take the active [1] class
    if len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
        shap_values = np.mean(shap_values, axis=0)[:, 1]
    # sometimes values are wrapped and require [0]
    elif len(shap_values.shape) and shap_values.shape[0] == 1:
        shap_values = shap_values[0]
    # if multiple inputs provided then average the importance across predictions
    elif len(shap_values.shape) > 1:
        shap_values = np.mean(shap_values, axis=0)

    # Handle important features mapping
    if important_feats is not None:
        sv = np.full(X_pred.shape[1], np.nan)
        sv[important_feats] = shap_values
        shap_values = sv

    # Initialize explainability dataframe
    exp_df = pd.DataFrame(
        {
            "shap_value": shap_values,
            "descriptor": np.nan,
            "bit": np.nan,
            "info": np.nan,
        }
    )

    # process single descriptors
    if descriptor_.name != "CompositeDescriptor":
        descriptor_name = descriptor_.name
        if descriptor_name == "ScaledDescriptor":
            descriptor_.name += f"_{descriptor_.parameters.descriptor.name}"
        exp_df["descriptor"] = descriptor_name
        exp_df["bit"] = range(len(shap_values))
        get_fp_info(exp_df, estimator, descriptor_, len(shap_values))
    # process CompositeDescriptor
    else:
        strt_idx = 0
        for descript in descriptor_.parameters.descriptors:
            fp_idx = descriptor_.fp_info()[json.dumps(serialize(descript))]
            if descript.name == "ScaledDescriptor":
                descript.name += f"_{descript.parameters.descriptor.name}"
            exp_df.loc[strt_idx : strt_idx + fp_idx - 1, "descriptor"] = descript.name
            exp_df.loc[strt_idx : strt_idx + fp_idx - 1, "bit"] = list(
                range(1, fp_idx + 1)
            )
            get_fp_info(exp_df, estimator, descript, fp_idx, strt_idx)
            strt_idx += fp_idx

    # Drop NaN values for important features
    if important_feats is not None:
        exp_df.dropna(subset=["shap_value"], inplace=True)

    return exp_df.sort_values("shap_value", ascending=False)


def subsetInputs(X_pred, estimator, max_results=25):
    """Subset the inputs to reduce the inference set size for explainability, based on both
    stratification of prediction scale and chemical diversity within stratified bins.

    This is particularly useful for expensive models like TabPFN which are costly at inference.

    Returns the subset of inputs that are representative of the original data with a fraction of
    inference cost.
    """
    # create predictions for X_pred
    if is_classifier(estimator):
        preds = estimator.predict_proba(X_pred)[:, 1]
    else:
        preds = estimator.predict(X_pred)

    # enumerate across the stratified bins
    strat = StratifiedOverlappingGroupKFold(fraction=max_results / len(X_pred))
    return next(strat.split(X_pred, preds))


def ExplainPreds(estimator, X_pred, mode, descriptor, n_cores=-1, subset="auto"):
    """Explain predictions using either SHAP (shallow models) or ChemProp interpret"""
    if hasattr(estimator, "interpret"):
        n_cores = effective_n_jobs(n_cores)
        return (
            estimator.interpret(X_pred)
            if n_cores == 1
            else pd.concat(
                Parallel(n_jobs=n_cores)(
                    delayed(estimator.interpret)([X]) for X in X_pred
                )
            )
        )

    X_pred = np.array(X_pred).astype(np.float64)
    if subset == "auto" and hasattr(estimator, "model_"):
        X_pred = X_pred[subsetInputs(X_pred, estimator)[-1]]
    if subset == "subsampled":
        X_pred = X_pred[subsetInputs(X_pred, estimator)[-1]]
    return ShapExplainer(estimator, X_pred, mode, descriptor)
