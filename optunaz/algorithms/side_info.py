import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, SVC


def process_side_info(si, y=None, rfe=False):
    """Perform variance threshold and co-correlated feature selection filtering.
    If si is regression then feature scaling and mad filtering is performed"""
    mod = None
    si = VarianceThreshold().fit_transform(si)
    si = pd.DataFrame(si)
    try:
        mod = unique_labels(si.fillna(0))
    except ValueError:
        scaler = MinMaxScaler(feature_range=(y.min(), y.max()))
        scaler.fit(si)
        si = pd.DataFrame(scaler.transform(si))
        mad = (si - si.mean()).abs().mean()
        mad_filter = mad > 0.1
        si = si[mad_filter[mad_filter].index]
    df_corr = si.corr(method="pearson", min_periods=1)
    df_not_correlated = ~(
        df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > 0.9
    ).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated].index
    si = si[un_corr_idx].to_numpy()
    if rfe:
        if mod is not None:
            svm = SVC(kernel="linear")
        else:
            svm = SVR(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=int(si.shape[1] * 0.1), step=1)
        rfe = rfe.fit(si, y)
        si = si[:, rfe.support_]
    return si


def binarise_side_info(si, cls=False):
    si = pd.DataFrame(si)
    si_nans = si.isna()
    cat_mask = ~si.apply(lambda s: pd.to_numeric(s, errors="coerce").notnull().all())
    si.loc[:, cat_mask] = (
        si.loc[:, cat_mask].astype("category").apply(lambda x: x.cat.codes)
    )
    if cls:
        si = (si > si.mean()).astype(np.uint8)
        si[si_nans] = np.nan
        return si.to_numpy()
    else:
        si[si_nans] = np.nan
        return si.astype(float).to_numpy()
