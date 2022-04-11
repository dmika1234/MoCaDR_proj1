from src.algorithm_functions import *
from tqdm import tqdm

os.chdir('D:\Studia\MoCaDR_proj1')

rep_vec = np.arange(20) + 1
lambda_vec = np.array([0.011, 0.0099])
eta_vec = np.array([0.001, 0.004, 0.007])
r_vec = np.array([2, 4, 7, 9, 16])
combs = np.array(np.meshgrid(rep_vec, lambda_vec, eta_vec, r_vec)).T.reshape(-1, 4)
results = pd.DataFrame(combs)
results['RMSE'] = 0.0
results.columns = ['rep', 'lambda', 'eta', 'r', 'RMSE']
results = results.reset_index()

for ix, row in tqdm(results.iterrows()):
    train_df, test_array = split_ratings('Datasets/ratings.csv')
    global_mean = train_df.mean().mean()
    n, d = train_df.shape
    W = np.matrix(np.full((n, int(row['r'])), np.sqrt(global_mean) / np.sqrt(row['r'])), dtype=np.longdouble)
    H = np.matrix(np.full((int(row['r']), d), np.sqrt(global_mean) / np.sqrt(int(row['r']))), dtype=np.longdouble)
    results.loc[ix, 'RMSE'] = perform_sgd(train_df, test_array, init_W=W, init_H=H, r=int(row['r']), lada=row['lambda'],
                                          learning_rate=row['eta'], max_iter=10000, min_diff=5e-10)[0]
    results.to_csv('Results/results_sgd_reps.csv', index=False)