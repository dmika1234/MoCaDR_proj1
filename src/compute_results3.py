from src.algorithm_functions import *
import random

train_df, test_array = split_ratings('Datasets/ratings.csv')
train_array = np.array(train_df)

lada = 0.1
eta = 0.1
r = 20
n, d = train_array.shape
W = np.matrix(np.full((n, r), 0), dtype='int64')
H = np.matrix(np.full((r, d), 0), dtype='int64')
possible_ix = np.argwhere(train_df.notnull().values).tolist()

for rep in np.arange(10000):
    ix = random.sample(range(0, len(possible_ix)), 1)
    i, j = possible_ix[ix[0]]
    grad_w = 2 * ((train_array[i, j] - float(W[i, :] * H[:, j])) * H[:, j]).T + 2 * lada * W[i, :]
    grad_h = 2 * ((train_array[i, j] - float(W[i, :] * H[:, j])) * W[i, :]).T + 2 * lada * H[:, j]
    W[i, :] = W[i, :] - eta * grad_w
    H[:, j] = H[:, j] - eta * grad_h

Z_tilde = np.array(np.dot(W, H))
print(calc_rmse(test_array, Z_tilde))
