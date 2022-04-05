from src.algorithm_functions import *
from tqdm import tqdm
import argparse


os.chdir('D:\Studia\MoCaDR_proj1')
np.random.seed(2022)
train_df, test_array = split_ratings('Datasets/ratings.csv')

# py -m src.compute_results_svd2 --min_r 1 --max_r 2
def ParseArguments():
    parser = argparse.ArgumentParser(description="Computing results for different methods and different r")
    parser.add_argument('--num_iter', default="1", required=False, help='We compute results from 1 to max_r dimensions')
    parser.add_argument('--min_r', default="1", required=False, help='We compute results from 1 to max_r dimensions')
    parser.add_argument('--max_r', default="1", required=False, help='We compute results from 1 to max_r dimensions')
    args = parser.parse_args()
    return args.min_r, args.max_r, args.n_reps


num_iter, min_r, max_r = ParseArguments()
num_iter = int(num_iter)
r_vec = np.arange(int(min_r), int(max_r) + 1)

results = pd.DataFrame({'iter': np.tile(np.arange(1, num_iter + 1), len(r_vec)),
                        'rmse': np.zeros(len(r_vec) * num_iter),
                        'r': np.repeat(r_vec, num_iter)})
results.to_csv('Results/results_stop_svd2', index=False)

train_array = fillna_row_means(train_df)

for ix, r in enumerate(tqdm(r_vec)):
    res = perform_svd2_test(train_df.isna(), train_array, test_array, r=r)
    results.loc[results['r'] == 3, 'rmse'] = res['rmse'].values


results.to_csv('Results/results_stop_svd2', index=False)
