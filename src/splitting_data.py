import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('../Datasets/ratings.csv')
train_df, test_df = train_test_split(data, train_size=0.90, stratify=data['userId'])
train_df.drop('Unnamed: 0', inplace=True, axis=1)
test_df.drop('Unnamed: 0', inplace=True, axis=1)
train_df.to_csv('../Datasets/sample_train.csv')
test_df.to_csv('../Datasets/sample_test.csv')
