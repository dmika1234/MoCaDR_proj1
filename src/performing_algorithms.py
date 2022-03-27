from src.algorithm_functions import *
import pandas as pd
import numpy as np

# Loading data
train_df = pd.read_csv('../Datasets/train_df.csv')
test_array = np.loadtxt('../Datasets/test_array.csv', delimiter=',')


# # Imputing with 0
# results = perform_svd1(fillna_value(train_df, 0), test_array, 10)
#
# # Imputing with global mean
# global_mean = train_df.mean().mean()
# results2 = perform_svd1(fillna_value(train_df, global_mean), test_array, 10)
#
# # Imputing with column means
#
#
# # Imputing with row means
# results4 = perform_svd1(fillna_row_means(train_df), test_array, 10)
#
# # Imputing with combined row and column means
# results5 = perform_svd1(fillna_means_combined(train_df, global_mean), test_array, 20)

# Imputing with weighted row and column means
# train_array_weighted = fillna_means_weighted(train_df, 1/2)
# results6 = perform_svd1(train_array_weighted, test_array, r=16)

# max_r = 300
# results6_diffR = pd.DataFrame({'r': np.arange(max_r) + 1, 'RMSE': np.zeros(max_r)})
# for i in np.arange(max_r):
#     results6_diffR['RMSE'][i] = perform_svd1(train_array_weighted, test_array, i)
#     print(i)
#
# results6_diffR.to_csv('../Results/rmse_diffr_svd1_weighted.csv', index=False)

# NMF
# results7 = perform_nmf(train_array_weighted, test_array, r=16)

# Finding best weigths for imputation
col_weight_arr = np.arange(0, 1 + 0.01, 0.01)
results_diff_weight = pd.DataFrame({'Column weight': col_weight_arr, 'RMSE': np.zeros(col_weight_arr.size)})
for i, weight in enumerate(col_weight_arr):
    train_array_weighted = fillna_means_weighted(train_df, weight)
    results_diff_weight['RMSE'][i] = perform_svd1(train_array_weighted, test_array, r=16)
    print(i)

results_diff_weight.to_csv('../Results/rmse_diff_weight_svd1_weighted.csv', index=False)

# print('0: ', results)
# print('Global mean: ', results2)
# print('Column means: ', results3)
# print('Row means: ', results4)
# print('Combined means: ', results5)
# print('SVD: ', results6)
# print('NMF: ', results7)
# print(results6_diffR)
