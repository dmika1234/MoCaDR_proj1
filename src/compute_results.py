import src.algorithm_functions as algs
from src.algorithm_functions import *
from tqdm import tqdm
import argparse


os.chdir('D:\Studia\MoCaDR_proj1')


def ParseArguments():
    parser = argparse.ArgumentParser(description="Computing results for different methods and different r")
    parser.add_argument('--max_r', default="1", required=False, help='We compute results from 1 to max_r dimensions')
    parser.add_argument('--alg', default="svd1", required=False, help='We compute results from 1 to max_r dimensions')
    args = parser.parse_args()

    return args.max_r, args.alg


max_r, alg = ParseArguments()
func_name = 'perform_' + alg
func_to_call = getattr(algs, func_name)

# Loading and splitting data
train_df, test_array = split_ratings('Datasets/ratings.csv')
# DataFrames for storing results
max_r = int(max_r)
r_vec = np.arange(max_r) + 1
results = pd.DataFrame({'r': r_vec, 'rmse_0': np.zeros(max_r), 'rmse_global_mean': np.zeros(max_r),
                             'rmse_row_means': np.zeros(max_r), 'rmse_means_weighted': np.zeros(max_r)})
results.to_csv('Results/results_' + alg + '.csv', index=False)


train_0 = fillna_value(train_df, 0)
global_mean = train_df.mean().mean()
train_global_mean = fillna_value(train_df, global_mean)
train_row_means = fillna_row_means(train_df)
train_means_weighted = fillna_means_weighted(train_df, 0.4)


if (func_name == 'perform_svd1') or (func_name == 'perform_nmf'):
    for ix, r in enumerate(tqdm(r_vec)):
        if r % 10 == 1:
            pd.read_csv('Results/results_' + alg + '.csv')
        results.loc[ix, 'rmse_0'] = func_to_call(train_0, test_array, r=r)
        results.loc[ix, 'rmse_global_mean'] = func_to_call(train_global_mean, test_array, r=r)
        results.loc[ix, 'rmse_row_means'] = func_to_call(train_row_means, test_array, r=r)
        results.loc[ix, 'rmse_means_weighted'] = func_to_call(train_means_weighted, test_array, r=r)
        if r % 10 == 0:
            results.to_csv('Results/results_' + alg + '.csv', index=False)

if func_name == 'perform_svd2':
    for ix, r in enumerate(tqdm(r_vec)):
        if r % 10 == 1:
            pd.read_csv('Results/results_' + alg + '.csv')
        results.loc[ix, 'rmse_0'] = func_to_call(train_df.isna(),
                                                 train_0, test_array, r=r)[0]
        results.loc[ix, 'rmse_global_mean'] = func_to_call(train_df.isna(),
                                                           train_global_mean, test_array, r=r)[0]
        results.loc[ix, 'rmse_row_means'] = func_to_call(train_df.isna(),
                                                         train_row_means, test_array, r=r)[0]
        results.loc[ix, 'rmse_means_weighted'] = func_to_call(train_df.isna(),
                                                              train_means_weighted, test_array, r=r)[0]
        if r % 10 == 0:
            results.to_csv('Results/results_' + alg + '.csv', index=False)


results.to_csv('Results/results_' + alg + '.csv', index=False)
