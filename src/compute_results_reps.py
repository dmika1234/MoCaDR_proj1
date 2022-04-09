from src.algorithm_functions import *
from tqdm import tqdm
import argparse


os.chdir('D:\Studia\MoCaDR_proj1')

# py -m src.compute_results --max_r 300 --alg nmf
def ParseArguments():
    parser = argparse.ArgumentParser(description="Computing results for different methods and different r")
    parser.add_argument('--alg', required=True, help='Algorithm')
    parser.add_argument('--nr_reps', required=True)
    args = parser.parse_args()

    return args.alg, args.nr_reps


alg, nr_reps = ParseArguments()
np.random.seed(2021)
reps_vec = np.arange(int(nr_reps))


# SVD1
if alg == 'svd1':
    param_vec = np.array([[0.39, 10], [0.38, 10], [0.42, 10], [0.36, 10], [0.39, 11],
                          [0.37, 10], [0.41, 10], [0.40, 10], [0.41, 9], [0.41, 13]])
    results = pd.DataFrame({'rep': np.repeat(reps_vec + 1, len(param_vec)),
                            'weight': np.tile(param_vec[:, 0], len(reps_vec)),
                            'r': np.tile(param_vec[:, 1], len(reps_vec)),
                            'RMSE': np.zeros(len(param_vec) * len(reps_vec))})
    ix = 0
    for rep in tqdm(reps_vec):
        train_df, test_array = split_ratings('Datasets/ratings.csv')
        for weight, r in tqdm(param_vec, leave=False):
            train_array = fillna_means_weighted(train_df, weight)
            results.loc[ix, 'RMSE'] = perform_svd1(train_array, test_array, r=int(r))
            ix += 1
    results.to_csv('Results/results_reps_svd1.csv', index=False)

# SVD2
if alg == 'svd2':
    param_vec = np.array([[0.25, 8], [0.26, 8], [0.24, 8], [0.27, 8], [0.28, 8],
                          [0.23, 8], [0.22, 8], [0.29, 8], [0.21, 8], [0.30, 8]])
    results = pd.DataFrame({'rep': np.repeat(reps_vec + 1, len(param_vec)),
                            'weight': np.tile(param_vec[:, 0], len(reps_vec)),
                            'r': np.tile(param_vec[:, 1], len(reps_vec)),
                            'RMSE': np.zeros(len(param_vec) * len(reps_vec))})
    ix = 0
    for rep in tqdm(reps_vec):
        train_df, test_array = split_ratings('Datasets/ratings.csv')
        for weight, r in tqdm(param_vec, leave=False):
            train_array = fillna_means_weighted(train_df, weight)
            results.loc[ix, 'RMSE'] = perform_svd2(train_df.isna(), train_array, test_array, r=int(r), min_diff=0.0086)[0]
            ix += 1
    results.to_csv('Results/results_reps_svd2.csv', index=False)

# NMF
if alg == 'nmf':
    param_vec = np.array([[0.40, 37], [0.41, 37], [0.39, 18], [0.39, 37], [0.40, 18],
                          [0.38, 18], [0.42, 37], [0.41, 18], [0.37, 18], [0.38, 37]])
    results = pd.DataFrame({'rep': np.repeat(reps_vec + 1, len(param_vec)),
                            'weight': np.tile(param_vec[:, 0], len(reps_vec)),
                            'r': np.tile(param_vec[:, 1], len(reps_vec)),
                            'RMSE': np.zeros(len(param_vec) * len(reps_vec))})
    ix = 0
    for rep in tqdm(reps_vec):
        train_df, test_array = split_ratings('Datasets/ratings.csv')
        for weight, r in tqdm(param_vec, leave=False):
            train_array = fillna_means_weighted(train_df, weight)
            results.loc[ix, 'RMSE'] = perform_nmf(train_array, test_array, r=int(r))
            ix += 1
    results.to_csv('Results/results_reps_nmf.csv', index=False)
