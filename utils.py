import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Utils:

    def labeling(self, df):
        # Data Labeling
        columns = df.columns.tolist()
        labeled_df = pd.DataFrame(columns=columns)
        self.mapping = []

        for column in columns:
            unique = df[column].unique()
            mapping = {}
            for index, val in enumerate(unique):
                mapping[val] = index
            self.mapping.append({column : mapping})
            labeled_df[column] = df[column].map(mapping)
        return labeled_df

    def one_hot_encoding(self, df: pd.DataFrame, column):
        one_hot_df = pd.get_dummies(df, columns=column)
        return one_hot_df

def test_train_split(percentage, X, Y):
    sample_size = int(X.shape[0] / 100) * percentage
    test_indices = np.random.randint(X.shape[0], size=sample_size)
    remaining_indices = np.delete(np.arange(X.shape[0]), test_indices)

    xtest = X[test_indices]
    ytest = Y[test_indices]

    xtrain = X[remaining_indices]
    ytrain = Y[remaining_indices]
    return xtest, ytest, xtrain, ytrain, test_indices, remaining_indices 

def precision_recall(ypredict, ytest):
    data = {'High': [0, 0, 0], 'Low': [0, 0, 0], 'Moderate': [0, 0, 0]}
    df = pd.DataFrame(data)
    for index in range(ypredict.shape[0]):
        if ypredict[index] == ytest[index]:
            val = ypredict[index]
            val = int(val)
            df.iloc[2-val, 2-val] = df.iloc[2-val, 2-val] + 1
        else:
            val = ytest[index]
            val = int(val)
            if ypredict[index] == 0:
                df.iloc[2,2-val] = df.iloc[2,2-val] + 1
            elif ypredict[index] == 1:
                df.iloc[1, 2-val] = df.iloc[1, 2-val] + 1
            else:
                df.iloc[0, 2-val] = df.iloc[0, 2-val] + 1
    return df