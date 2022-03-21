import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


# Functions for algorithms
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


# Functions for imputation
def fillna_value(dataframe, value: float) -> np.ndarray:
    return np.array(dataframe.fillna(value))


def fillna_row_means(dataframe) -> np.ndarray:
    return np.array(dataframe.T.fillna(dataframe.mean(axis=1)).T)


# Imputing with column means
# train_array3 = train_df.fillna(train_df.mean())
# train_array3 = np.array(train_array3.T.fillna(train_array3.mean(axis=1)).T)
# results3 = perform_svd1(train_array3, test_array, 10)


def fillna_means_combined(dataframe, value: float) -> np.ndarray:
    col_means = np.matrix(dataframe.mean().values)
    row_means = np.matrix(dataframe.mean(axis=1).values).T
    #   Creating custom values to fill NaN in train_df
    fill_matrix = np.dot(row_means, col_means / value)
    fill_df = pd.DataFrame(fill_matrix)
    fill_df = fill_df.T.fillna(fill_df.mean(axis=1)).T
    fill_array = np.array(fill_df)
    na_indx = dataframe.isna()
    train_array = np.array(dataframe)
    train_array[na_indx] = np.array(fill_array[na_indx]).reshape(-1)
    return train_array

def fillna_means_weighted(dataframe, column_weight: float, row_weight: float) -> np.ndarray:
    col_means = np.matrix(dataframe.mean().values)
    row_means = np.matrix(dataframe.mean(axis=1).values).T
    fill_matrix = np.log(np.dot(np.exp(row_weight * row_means), np.exp(column_weight * col_means)))
    fill_df = pd.DataFrame(fill_matrix)
    fill_df = fill_df.T.fillna(fill_df.mean(axis=1)).T
    fill_array = np.array(fill_df)
    train_array = np.array(dataframe)
    na_indx = dataframe.isna()
    train_array[na_indx] = np.array(fill_array[na_indx]).reshape(-1)
    return train_array
