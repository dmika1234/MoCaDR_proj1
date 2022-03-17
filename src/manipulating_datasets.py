import numpy as np
import pandas as pd
import random as rng

# We don't need timestamp column
raw_df = pd.read_csv('../Datasets/ratings_dirt.csv')
df = raw_df.drop('timestamp', axis=1)
df.to_csv('Datasets/ratings.csv')

# Take some small subset of our data for testing purposes
rand_ix = rng.sample(list(np.unique(df['userId'].values)), 10)
sub_df = df.loc[df['userId'].isin(rand_ix)]
sub_df.to_csv('Datasets/small_ratings.csv')
