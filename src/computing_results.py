from src.algorithm_functions import *

os.chdir('D:\Studia\MoCaDR_proj1')

# Loading and splitting data
train_df, test_array = split_ratings('Datasets/ratings.csv')
# DataFrames for storing results
max_r = 2
r_vec = np.arange(max_r) + 1
results_svd1 = pd.DataFrame({'r': r_vec, 'rmse_0': np.zeros(max_r), 'rmse_global_mean': np.zeros(max_r),
                             'rmse_row_means': np.zeros(max_r), 'rmse_means_weighted': np.zeros(max_r)})
results_svd2 = pd.DataFrame({'r': r_vec, 'rmse_0': np.zeros(max_r), 'rmse_global_mean': np.zeros(max_r),
                             'rmse_row_means': np.zeros(max_r), 'rmse_means_weighted': np.zeros(max_r)})
results_nmf = pd.DataFrame({'r': r_vec, 'rmse_0': np.zeros(max_r), 'rmse_global_mean': np.zeros(max_r),
                            'rmse_row_means': np.zeros(max_r), 'rmse_means_weighted': np.zeros(max_r)})

train_0 = fillna_value(train_df, 0)
global_mean = train_df.mean().mean()
train_global_mean = fillna_value(train_df, global_mean)
train_row_means = fillna_row_means(train_df)
train_means_weighted = fillna_means_weighted(train_df, 0.4)

for ix, r in enumerate(r_vec):
    # SVD1
    results_svd1.loc[ix, 'rmse_0'] = perform_svd1(train_0, test_array, r=r)
    results_svd1.loc[ix, 'rmse_global_mean'] = perform_svd1(train_global_mean, test_array, r=r)
    results_svd1.loc[ix, 'rmse_row_means'] = perform_svd1(train_row_means, test_array, r=r)
    results_svd1.loc[ix, 'rmse_means_weighted'] = perform_svd1(train_means_weighted, test_array, r=r)
    # SVD2
    na_ix = train_df.isna()
    results_svd2.loc[ix, 'rmse_0'] = perform_svd2(na_ix, train_0, test_array, r=r, max_iter=100)[0]
    results_svd2.loc[ix, 'rmse_global_mean'] = perform_svd2(na_ix, train_global_mean, test_array, r=r, max_iter=100)[0]
    results_svd2.loc[ix, 'rmse_row_means'] = perform_svd2(na_ix, train_row_means, test_array, r=r, max_iter=100)[0]
    results_svd2.loc[ix, 'rmse_means_weighted'] = perform_svd2(na_ix, train_means_weighted, test_array,
                                                               r=r, max_iter=100)[0]
    # NMF
    results_nmf.loc[ix, 'rmse_0'] = perform_nmf(train_0, test_array, r=r)
    results_nmf.loc[ix, 'rmse_global_mean'] = perform_nmf(train_global_mean, test_array, r=r)
    results_nmf.loc[ix, 'rmse_row_means'] = perform_nmf(train_row_means, test_array, r=r)
    results_nmf.loc[ix, 'rmse_means_weighted'] = perform_nmf(train_means_weighted, test_array, r=r)

results_svd1.to_csv('Results/results_svd1.csv', index=False)
results_svd2.to_csv('Results/results_svd2.csv', index=False)
results_nmf.to_csv('Results/results_nmf.csv', index=False)
