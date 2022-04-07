from src.algorithm_functions import *
from tqdm import tqdm
os.chdir('../')


train_df, test_array = split_ratings('Datasets/ratings.csv')
train_array = fillna_means_weighted(train_df, 1/2)
stop_vec = np.arange(0.0069, 0.009 + 0.0001, 0.0001)
r_vec = np.concatenate((np.arange(5, 20), np.arange(20, 51, 5)))

results = pd.DataFrame({'stop': np.repeat(stop_vec, len(r_vec)),
                        'r': np.tile(r_vec, len(stop_vec)),
                            'RMSE': np.zeros(len(r_vec) * len(stop_vec))})
ix = 0
for stop in tqdm(stop_vec):
    for r in tqdm(r_vec, leave=False):
        results.loc[ix, 'RMSE'] = perform_svd2(train_df.isna(), train_array, test_array,
                                               r=r, min_diff=stop, max_iter=100)[0]
        ix += 1
results.to_csv('Results/results_stop_svd2.csv', index=False)

