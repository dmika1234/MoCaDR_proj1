import src.algorithm_functions as algs
from src.algorithm_functions import *
from tqdm import tqdm
import argparse



os.chdir('D:\Studia\MoCaDR_proj1')
np.random.seed(2022)

# py -m src.compute_results --min_r 1 --max_r 1 --n_reps 1
def ParseArguments():
    parser = argparse.ArgumentParser(description="Computing results for different methods and different r")
    parser.add_argument('--min_r', default="1", required=False, help='We compute results from 1 to max_r dimensions')
    parser.add_argument('--max_r', default="1", required=False, help='We compute results from 1 to max_r dimensions')
    parser.add_argument('--n_reps', default="1", required=False, help='We compute results from 1 to max_r dimensions')
    args = parser.parse_args()

    return args.min_r, args.max_r, args.n_reps


min_r, max_r, n_reps = ParseArguments()

r_vec = np.arange(int(min_r), int(max_r) + 1)
n_reps = int(n_reps)

results = pd.DataFrame({'r': np.tile(r_vec, n_reps), 'rep': np.repeat(np.arange(1, n_reps + 1), len(r_vec)),
                        'SVD1': np.zeros(len(r_vec) * n_reps),
                        'SVD2': np.zeros(len(r_vec) * n_reps),
                        'NMF': np.zeros(len(r_vec) * n_reps)})
results.to_csv('Results/results_reps', index=False)
ix = 0
for rep in tqdm(np.arange(n_reps) + 1):
    train_df, test_array = split_ratings('Datasets/ratings.csv')
    train_array_svd1 = fillna_means_weighted(train_df, 0.37)
    train_array_svd2 = fillna_means_weighted(train_df, 0.32)
    train_array_nmf = fillna_means_weighted(train_df, 0.4)
    for r in r_vec:
        results.loc[ix, 'SVD1'] = perform_svd1(train_array_svd1, test_array, r=r)
        results.loc[ix, 'SVD2'] = perform_svd2(train_df.isna(), train_array_svd2, test_array, r=r)[0]
        results.loc[ix, 'NMF'] = perform_nmf(train_array_nmf, test_array, r=r)
        ix += 1
    results.to_csv('Results/results_reps.csv', index=False)

results.to_csv('Results/results_reps.csv', index=False)
