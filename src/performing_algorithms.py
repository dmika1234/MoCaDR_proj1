import numpy as np
import pandas as pd
from src.splitting_data import train_df, test_df, data
from src.algorithm_functions import perform_svd1

train_set = train_df
test_set = test_df

# Reshape dataset to wide format to have it matrix-like
train_wide = train_set.pivot(index='userId', columns='movieId', values='rating')
test_wide = test_set.pivot(index='userId', columns='movieId', values='rating')
test_wide.sort_index(axis=1, inplace=True)
train_wide.sort_index(axis=1, inplace=True)

#            Rearranging training and test data sets to have columns from the whole data set

# Movie ids in the whole dataset
movies_ids = np.unique(data.movieId)
# Movie ids in the training dataset
movies_ids_train = np.unique(train_set.movieId)
# Movie ids that are not in training data
movies_ids_miss = np.setdiff1d(movies_ids, movies_ids_train)
# Adding columns with movies not in train set but which appear in the whole data set
if len(movies_ids_miss) != 0:
    nas_array = np.empty((train_wide.shape[0], movies_ids_miss.shape[0],))
    nas_array[:] = np.nan
    missing_df = pd.DataFrame(nas_array)
    missing_df.columns = movies_ids_miss
    missing_df.index = train_wide.index
    train_df = pd.concat([train_wide, missing_df], axis=1)
    # Sorting column names to control positions of movieId
    train_df.sort_index(axis=1, inplace=True)
else:
    pass

# Same for test dataset
movies_ids = np.unique(data.movieId)
movies_ids_test = np.unique(test_set.movieId)
movies_ids_miss2 = np.setdiff1d(movies_ids, movies_ids_test)
if len(movies_ids_miss2) != 0:
    nas_array2 = np.empty((test_wide.shape[0], movies_ids_miss2.shape[0],))
    nas_array2[:] = np.nan
    missing_df2 = pd.DataFrame(nas_array2)
    missing_df2.columns = movies_ids_miss2
    missing_df2.index = test_wide.index
    test_df = pd.concat([test_wide, missing_df2], axis=1)
    test_df.sort_index(axis=1, inplace=True)
    test_array = np.array(test_df)
else:
    test_array = test_wide





#             Imputation

# Imputing with 0
train_df.fillna(0)
train_array = np.array(train_df.fillna(0))

results = perform_svd1(train_array, test_array, 2)
print(results)
