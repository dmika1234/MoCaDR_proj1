from src.algorithm_functions import *
from tqdm import tqdm


os.chdir('D:\Studia\MoCaDR_proj1')


np.random.seed(2022)
train_df, test_array = split_ratings('Datasets/ratings.csv')

# Finding best weigths for imputation
col_weight_arr = np.arange(0, 1 + 0.01, 0.01)
results_diff_weight = pd.DataFrame({'Column weight': col_weight_arr, 'SVD1': np.zeros(col_weight_arr.size),
                                    'SVD2': np.zeros(col_weight_arr.size), 'NMF': np.zeros(col_weight_arr.size)})

for ix, weight in enumerate(tqdm(col_weight_arr)):
    train_array_weighted = fillna_means_weighted(train_df, weight)
    results_diff_weight.loc[ix, 'SVD1'] = perform_svd1(train_array_weighted, test_array, r=10)
    results_diff_weight.loc[ix, 'SVD2'] = perform_svd2(train_df.isna(), train_array_weighted, test_array, r=8)[0]
    results_diff_weight.loc[ix, 'NMF'] = perform_svd1(train_array_weighted, test_array, r=37)

results_diff_weight.to_csv('Results/results_weights.csv', index=False)
