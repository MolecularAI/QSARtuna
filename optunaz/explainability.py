import numpy as np
import copy
import pandas as pd
import json
from apischema import serialize
from optunaz.config import ModelMode

from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed, effective_n_jobs


def get_ecfp_fpinfo(m, descriptor):
    """Return the ecfp info for a compound mol"""

    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        m,
        radius=descriptor.parameters.radius,
        nBits=descriptor.parameters.nBits,
        bitInfo=info,
    )
    return info


def get_ecfpcount_fpinfo(m, descriptor):
    """Return the ecfp_count info for a compound mol"""

    info = {}
    fp = AllChem.GetHashedMorganFingerprint(
        m,
        radius=descriptor.parameters.radius,
        nBits=descriptor.parameters.nBits,
        useFeatures=descriptor.parameters.useFeatures,
        bitInfo=info,
    )
    return info


def explain_ECFP(len_feats, estimator, descriptor):
    """Explain ECFPs using train atom environments"""

    ret = np.empty(len_feats, dtype="<U50")
    # enumerate through each important feature
    for feat_idx in range(len_feats):
        this_feat_explained = False
        # enumerate through training set searching a compound with the feature
        for mol_idx, mol in enumerate(estimator.X_):
            if not this_feat_explained:
                # the feature is present in the mol if bit is > 0
                if mol[feat_idx] > 0:
                    m = Chem.MolFromSmiles(estimator.train_smiles_[mol_idx])
                    if descriptor.name == "ECFP":
                        info = get_ecfp_fpinfo(m, descriptor)[feat_idx]
                    elif descriptor.name == "ECFP_counts":
                        info = get_ecfpcount_fpinfo(m, descriptor)[feat_idx]
                    # enumerate atom matches, breaking when valid smiles produced
                    for atom, radius in info:
                        env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, atom)
                        amap = {}
                        submol = Chem.PathToSubmol(m, env, atomMap=amap)
                        try:
                            feat_smi = Chem.MolToSmiles(
                                submol, rootedAtAtom=amap[atom], canonical=False
                            )
                            ret[feat_idx] = feat_smi
                            this_feat_explained = True
                            # continue trying other matches if blank feature
                            if feat_smi != "":
                                break
                        # sometime MolToSmiles fails
                        except KeyError:
                            pass
            # feature is explained, so break
            else:
                break
    return ret


def get_fp_info(exp_df, estimator, descript, fp_idx, strt_idx=None):
    """Get ECFP SMILES environments or Physchem names when available"""
    info = []
    if "ECFP" in descript.name:
        info = explain_ECFP(fp_idx, estimator, descript)
    elif "Physchem" in descript.name:
        try:
            info = descript.parameters.rdkit_names
        except AttributeError:
            info = descript.parameters.descriptor.parameters.rdkit_names
    elif "Jazzy" in descript.name:
        try:
            info = descript.parameters.jazzy_names
        except AttributeError:
            info = descript.parameters.descriptor.parameters.jazzy_names
    if len(info) > 0:
        if strt_idx is not None:
            exp_df.loc[strt_idx : strt_idx + fp_idx - 1, "info"] = info
        else:
            exp_df["info"] = info
    return


def runShap(estimator, X_pred, mode):
    """Explain model prediction using auto explainer or SHAP KernelExplainer"""
    import shap

    # see if shap can auto explain
    try:
        try:
            explainer = shap.Explainer(estimator, estimator.X_)
        # deal with methods that require the inference method
        except TypeError:
            if mode == ModelMode.REGRESSION:
                explainer = shap.Explainer(estimator.predict, estimator.X_)
            if mode == ModelMode.CLASSIFICATION:
                explainer = shap.Explainer(estimator.predict_proba, estimator.X_)
        shap_values = np.abs(np.array(explainer.shap_values(np.array(X_pred))))
    # use kernel for other models
    except AttributeError:
        if mode == ModelMode.REGRESSION:
            explainer = shap.KernelExplainer(estimator.predict, estimator.X_)
        if mode == ModelMode.CLASSIFICATION:
            explainer = shap.KernelExplainer(estimator.predict_proba, estimator.X_)
        shap_values = np.abs(
            np.array(explainer.shap_values(np.array(X_pred[:1]), nsamples="auto"))
        )
    return shap_values


def ShapExplainer(estimator, X_pred, mode, descriptor):
    """
    Run SHAP and populate the explainability dataframe
    """
    shap_values = runShap(estimator, X_pred, mode)
    descriptor_ = copy.deepcopy(descriptor)

    # process the shap_values shapes
    if len(shap_values.shape) == 3:
        if shap_values.shape[0] == 2:
            # if explainer explains both classes, take the active [1] class
            shap_values = shap_values[1]
        elif shap_values.shape[0] == 1:
            # sometimes values are wrapped and require [0]
            shap_values = shap_values[0]
    # if multiple inputs provided then average the importance across predictions
    if len(shap_values.shape) > 1:
        shap_values = np.mean(shap_values, axis=0)

    exp_df = pd.DataFrame(
        data={
            "shap_value": shap_values,
            "descriptor": np.nan,
            "bit": np.nan,
            "info": np.nan,
        }
    )

    # process single descriptors
    if descriptor_.name != "CompositeDescriptor":
        if descriptor_.name == "ScaledDescriptor":
            descriptor_.name += f"_{descriptor_.parameters.descriptor.name}"
        exp_df["descriptor"] = descriptor_.name
        exp_df["bit"] = range(len(shap_values))
        get_fp_info(exp_df, estimator, descriptor_, len(shap_values))
    # process CompositeDescriptor
    else:
        fp_info = descriptor_.fp_info()
        strt_idx = 0
        info = None
        for descript in descriptor_.parameters.descriptors:
            fp_idx = fp_info[json.dumps(serialize(descript))]
            if descript.name == "ScaledDescriptor":
                descript.name += f"_{descript.parameters.descriptor.name}"
            for col, col_value in [
                ("descriptor", descript.name),
                ("bit", range(1, fp_idx + 1, 1)),
            ]:
                exp_df.loc[strt_idx : strt_idx + fp_idx - 1, col] = col_value
            get_fp_info(exp_df, estimator, descript, fp_idx, strt_idx=strt_idx)
            strt_idx += fp_idx
    return exp_df.sort_values("shap_value", ascending=False)


def ExplainPreds(estimator, X_pred, mode, descriptor):
    """Explain predictions using either SHAP (shallow models) or ChemProp interpret"""
    if hasattr(estimator, "interpret"):
        n_cores = effective_n_jobs(-1)
        if n_cores == 1:
            return estimator.interpret(X_pred)
        else:
            return pd.concat(
                Parallel(n_jobs=n_cores)(
                    delayed(estimator.interpret)([X]) for X in X_pred
                )
            )
    else:
        return ShapExplainer(estimator, X_pred, mode, descriptor)
