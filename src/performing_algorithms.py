from src.algorithm_functions import *


train_df = pd.read_csv('../Datasets/train_df.csv')
train_df.drop('userId', axis=1, inplace=True)
test_array = np.loadtxt('../Datasets/test_array.csv', delimiter=',')


# Imputing with 0
results = perform_svd1(fillna_value(train_df, 0), test_array, 10)

# Imputing with global mean
global_mean = train_df.mean().mean()
results2 = perform_svd1(fillna_value(train_df, global_mean), test_array, 10)

# Imputing with column means


# Imputing with row means
results4 = perform_svd1(fillna_row_means(train_df), test_array, 10)

# Imputing with combined row and column means
results5 = perform_svd1(fillna_means_combined(train_df, global_mean), test_array, 20)

# Imputing with weighted row and column means
results6 = perform_svd1(fillna_means_weighted(train_df, 1/2, 1/2), test_array, 20)

print('0: ', results)
print('Global mean: ', results2)
# print('Column means: ', results3)
print('Row means: ', results4)
print('Combined means: ', results5)
print('Weighted means: ', results6)
