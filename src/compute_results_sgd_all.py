from src.algorithm_functions import *
from tqdm import tqdm

os.chdir('D:\Studia\MoCaDR_proj1')
np.random.seed(2022)
train_df, test_array = split_ratings('Datasets/ratings.csv')
lambda_vec = np.concatenate((np.arange(0.001, 0.02, 0.001), np.arange(0.02, 0.051, 0.005)))
eta_vec = np.array([0.008, 0.009, 0.0099, 0.01, 0.011, 0.012, 0.02])
r_vec = np.concatenate((np.arange(1, 20), np.array([20, 25, 40, 50])))
combs = np.array(np.meshgrid(lambda_vec, eta_vec, r_vec)).T.reshape(-1, 3)
results = pd.DataFrame(combs)
results['RMSE'] = 0.0
results.columns = ['lambda', 'eta', 'r', 'RMSE']
results = results.reset_index()
global_mean = train_df.mean().mean()
n, d = train_df.shape

for ix, row in tqdm(results.iterrows()):

    W = np.matrix(np.full((n, int(row['r'])), np.sqrt(global_mean) / np.sqrt(row['r'])), dtype=np.longdouble)
    H = np.matrix(np.full((int(row['r']), d), np.sqrt(global_mean) / np.sqrt(int(row['r']))), dtype=np.longdouble)
    results.loc[ix, 'RMSE'] = perform_sgd(train_df, test_array, init_W=W, init_H=H, r=int(row['r']), lada=row['lambda'],
                                          learning_rate=row['eta'], max_iter=10000, min_diff=5e-10)[0]
    results.to_csv('Results/results_sgd_all2.csv', index=False)