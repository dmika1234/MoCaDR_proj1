import numpy as np
import pandas as pd
import copy
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF


os.chdir('../')


# Functions for algorithms
def perform_svd1(train_array: np.ndarray, test_array: np.ndarray, r: int) -> float:

    svd = TruncatedSVD(n_components=r)
    svd.fit(train_array)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(train_array) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_tilde = np.dot(W, H)
    # Calculating RMSE
    rmse = calc_rmse(test_array, Z_tilde)

    return rmse


# na_indx = train_df.isna()
def perform_svd2(na_indx, train_array: np.ndarray, test_array: np.ndarray, r: int,
                      max_iter: int = 100, min_diff: float = 0.0089) -> tuple:

    Z_i = copy.deepcopy(train_array)
    m = copy.deepcopy(train_array[~na_indx])
    i = 0
    diff = 10 ** 5

    while (i < max_iter) & (min_diff < diff):
        Z_i[~na_indx] = np.array(m).reshape(-1)
        rmse_i = calc_rmse(test_array, Z_i)
        svd = TruncatedSVD(n_components=r)
        svd.fit(Z_i)
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W = svd.transform(train_array) / svd.singular_values_
        H = np.dot(Sigma2, VT)
        Z_ii = np.dot(W, H)
        diff = ((Z_ii - Z_i) ** 2).sum() / (Z_ii.shape[0] * Z_ii.shape[1])
        i += 1
        rmse = calc_rmse(test_array, Z_ii)
        Z_i = copy.deepcopy(Z_ii)

    return rmse, i


def perform_nmf(train_array: np.ndarray, test_array: np.ndarray, r: int, random_state: int = 0) -> float:

    model = NMF(n_components=r, init='random', random_state=random_state, max_iter=200)
    W = model.fit_transform(train_array)
    H = model.components_
    Z_tilde = np.dot(W, H)
    # Calculating RMSE
    rmse = calc_rmse(test_array, Z_tilde)

    return rmse


def perform_sgd(train_array, test_array, r: int, init_vec, eta: float, alpha: float):
    n, d = train_array.shape
    W = np.zeros([n, r])
    H = np.zeros([r, d])

    while True:
        E = train_array - np.dot(W, H)




# Function to calculate RMSE
def calc_rmse(test_array: np.ndarray, estimated_array: np.ndarray) -> float:
    test_array  = np.array(test_array, dtype=np.longdouble)
    estimated_array = np.array(estimated_array, dtype=np.longdouble)
    diff = test_array - estimated_array
    num_of_vals = (~np.isnan(diff)).sum()
    # Delete not NaN values to further summing
    diff = np.array(diff[~np.isnan(diff)], dtype=np.longdouble)
    rmse = np.sqrt(1 / np.abs(num_of_vals) * ((diff ** 2).sum()))

    return rmse


# Functions for imputation
def fillna_value(dataframe, value: float) -> np.ndarray:
    return np.array(dataframe.fillna(value))


def fillna_row_means(dataframe) -> np.ndarray:
    return np.array(dataframe.T.fillna(dataframe.mean(axis=1)).T)


def fillna_col_means(dataframe) -> np.ndarray:
    glob_mean = dataframe.mean().mean()
    col_means  = dataframe.mean()
    col_means[col_means.isna()] = glob_mean
    return np.array(dataframe.fillna(col_means))


def fillna_means_combined(dataframe) -> np.ndarray:
    global_mean = dataframe.mean().mean()
    col_means = np.matrix(dataframe.mean().values)
    row_means = np.matrix(dataframe.mean(axis=1).values).T
    #   Creating custom values to fill NaN in train_df
    fill_matrix = np.dot(row_means, col_means / global_mean)
    fill_matrix = np.minimum(fill_matrix, 5)
    fill_df = pd.DataFrame(fill_matrix)
    fill_df = fill_df.T.fillna(fill_df.mean(axis=1)).T
    fill_array = np.array(fill_df)
    na_indx = dataframe.isna()
    train_array = np.array(dataframe)
    train_array[na_indx] = np.array(fill_array[na_indx]).reshape(-1)
    return train_array


def fillna_means_weighted(dataframe, column_weight: float) -> np.ndarray:
    row_weight = 1 - column_weight
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


def split_ratings(filepath: str, train_size: float = 0.9) -> tuple:
    data = pd.read_csv(filepath)
    data = data[['userId', 'movieId', 'rating']]
    train_df, test_df = train_test_split(data, train_size=train_size, stratify=data['userId'])

    train_set = train_df
    test_set = test_df

    # Reshape dataset to wide format to have it matrix-like
    train_wide = train_set.pivot(index='userId', columns='movieId', values='rating')
    test_wide = test_set.pivot(index='userId', columns='movieId', values='rating')
    test_wide.sort_index(axis=1, inplace=True)
    train_wide.sort_index(axis=1, inplace=True)

    #            Rearranging training and test data sets to have columns from the whole data set

    # Movie ids in the whole dataset
    movies_ids = np.unique(data.movieId)
    # Movie ids in the training dataset
    movies_ids_train = np.unique(train_set.movieId)
    # Movie ids that are not in training data
    movies_ids_miss = np.setdiff1d(movies_ids, movies_ids_train)
    # Adding columns with movies not in train set but which appear in the whole data set
    if len(movies_ids_miss) != 0:
        nas_array = np.empty((train_wide.shape[0], movies_ids_miss.shape[0],))
        nas_array[:] = np.nan
        missing_df = pd.DataFrame(nas_array)
        missing_df.columns = movies_ids_miss
        missing_df.index = train_wide.index
        train_df = pd.concat([train_wide, missing_df], axis=1)
        # Sorting column names to control positions of movieId
        train_df.sort_index(axis=1, inplace=True)
    else:
        pass

    # Same for test dataset
    movies_ids = np.unique(data.movieId)
    movies_ids_test = np.unique(test_set.movieId)
    movies_ids_miss2 = np.setdiff1d(movies_ids, movies_ids_test)
    if len(movies_ids_miss2) != 0:
        nas_array2 = np.empty((test_wide.shape[0], movies_ids_miss2.shape[0],))
        nas_array2[:] = np.nan
        missing_df2 = pd.DataFrame(nas_array2)
        missing_df2.columns = movies_ids_miss2
        missing_df2.index = test_wide.index
        test_df = pd.concat([test_wide, missing_df2], axis=1)
        test_df.sort_index(axis=1, inplace=True)
        test_array = np.array(test_df)
    else:
        test_array = np.array(test_wide)

    return train_df, test_array
