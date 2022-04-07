from src.algorithm_functions import *
from tqdm import tqdm
import argparse


os.chdir('../')

# py -m src.compute_results --max_r 300 --alg nmf
def ParseArguments():
    parser = argparse.ArgumentParser(description="Computing results for different methods and different r")
    parser.add_argument('--alg', required=True, help='Algorithm')
    args = parser.parse_args()

    return args.alg


alg = ParseArguments()
# Loading and splitting data
np.random.seed(2022)
train_df, test_array = split_ratings('Datasets/ratings.csv')
weights_vec = np.arange(0.2, 0.6 + 0.01, 0.01)


# SVD1
if alg == 'svd1':
    r_vec = np.concatenate((np.arange(5, 20), np.arange(20, 51, 5)))
    results = pd.DataFrame({'col_weight': np.repeat(weights_vec, len(r_vec)),
                            'r': np.tile(r_vec, len(weights_vec)),
                            'RMSE': np.zeros(len(r_vec) * len(weights_vec))})
    ix = 0
    for weight in tqdm(weights_vec):
        train_array = fillna_means_weighted(train_df, weight)
        for r in tqdm(r_vec, leave=False):
            results.loc[ix, 'RMSE'] = perform_svd1(train_array, test_array, r=r)
            ix += 1
    results.to_csv('Results/results_r_w_svd1.csv', index=False)

# SVD2
if alg == 'svd2':
    r_vec = np.concatenate((np.arange(5, 20), np.arange(20, 51, 5)))
    results = pd.DataFrame({'col_weight': np.repeat(weights_vec, len(r_vec)),
                            'r': np.tile(r_vec, len(weights_vec)),
                            'RMSE': np.zeros(len(r_vec) * len(weights_vec))})
    ix = 0
    for weight in tqdm(weights_vec):
        train_array = fillna_means_weighted(train_df, weight)
        for r in tqdm(r_vec, leave=False):
            results.loc[ix, 'RMSE'] = perform_svd2(train_df.isna(), train_array, test_array, r=r)[0]
            ix += 1
    results.to_csv('Results/results_r_w_svd2.csv', index=False)

# NMF
if alg == 'svd2':
    r_vec = np.concatenate((np.arange(5, 20), np.arange(20, 51, 5)))
    results = pd.DataFrame({'col_weight': np.repeat(weights_vec, len(r_vec)),
                            'r': np.tile(r_vec, len(weights_vec)),
                            'RMSE': np.zeros(len(r_vec) * len(weights_vec))})
    ix = 0
    for weight in tqdm(weights_vec):
        train_array = fillna_means_weighted(train_df, weight)
        for r in tqdm(r_vec, leave=False):
            results.loc[ix, 'RMSE'] = perform_nmf(train_array, test_array, r=r)
            ix += 1
    results.to_csv('Results/results_r_w_nmf.csv', index=False)
