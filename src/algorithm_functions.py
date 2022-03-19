import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def perform_svd1(train_array: np.ndarray, test_array: np.ndarray, n_components: int) -> float:

    svd = TruncatedSVD(n_components=n_components)
    svd.fit(train_array)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(train_array) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_tilde = np.dot(W, H)

    # Calculating RMSE
    diff = test_array - Z_tilde
    num_of_vals = (~np.isnan(diff)).sum()
    # Delete not NaN values to further summing
    diff = diff[~np.isnan(diff)]
    rmse = np.sqrt(1 / np.abs(num_of_vals) * ((diff ** 2).sum()))

    return rmse


