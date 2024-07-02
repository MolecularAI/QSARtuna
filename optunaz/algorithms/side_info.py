import requests
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, SVC


def process_side_info(si, y=None):
    """Perform variance threshold and co-correlated feature selection filtering.
    If si is regression then feature scaling and mad filtering is performed"""
    sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
    si = sel.fit_transform(si)
    try:
        mod = unique_labels(si)
        si = pd.DataFrame(si)
    except ValueError:
        mod = None
        scaler = MinMaxScaler()
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
    if y is not None:
        if mod is not None:
            svm = SVR(kernel="linear")
        else:
            svm = SVC(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=int(si.shape[1] * 0.1), step=1)
        rfe = rfe.fit(si, y)
        si = si[:, rfe.support_]
    return si


def binarise_side_info(si):
    means = np.mean(si, axis=0)
    return np.array(si > means, dtype=np.uint8)
