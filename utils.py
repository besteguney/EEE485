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

def chi2_square(df, col):
    contingency_table = pd.crosstab(df[col], df['Adaptivity Level'])
    contingency_table.loc['total'] = contingency_table.sum(axis=0)
    contingency_table['total'] = contingency_table.sum(axis=1)

    ## calculating for expected value
    expected_val = contingency_table.copy()
    for row_index in range(contingency_table.shape[0] - 1):
        for column_index in range(contingency_table.shape[1] - 1):
            total_1 = contingency_table.loc['total', column_index] 
            total_2 = contingency_table.loc[row_index, 'total'] 
            expected_val.iloc[row_index, column_index] = (total_1 * total_2) / (total_1 + total_2)
    expected_val.loc['total'] = expected_val.sum(axis=0) - contingency_table.loc['total']
    expected_val['total'] = expected_val.sum(axis=1) - contingency_table['total']

    x2_table = contingency_table.copy()
    for row_index in range(contingency_table.shape[0] - 1):
        for column_index in range(contingency_table.shape[1] - 1):
            x2_table.iloc[row_index, column_index] = (contingency_table.iloc[row_index, column_index] - expected_val.iloc[row_index, column_index]) ** 2
            x2_table.iloc[row_index, column_index] = x2_table.iloc[row_index, column_index] / expected_val.iloc[row_index, column_index]
    x2_table.loc['total'] = x2_table.sum(axis=0) - contingency_table.loc['total']
    x2_table['total'] = x2_table.sum(axis=1) - contingency_table['total']

    return x2_table.loc['total', 'total'], (contingency_table.shape[0] - 2) * (contingency_table.shape[1] - 2)

## Feature ranking with mutual information
def find_mutual(X, feature_index, Y):
    return mutual_information(X[:, feature_index], Y)

def mutual_information(X, Y):
    features = np.unique(X)
    labels = np.unique(Y)
    sum = 0

    for label in labels:
        indices = np.where(Y == label)[0]
        cur_y = Y[indices]
        cur_x = X[indices]
        for feature in features:
            x_indices = np.where(cur_x == feature)[0]
            cur = cur_x[x_indices] 
            joint_probability = len(cur) / len(Y)
            if joint_probability == 0:
                continue
            
            feature_indices = np.where(X == feature)[0]
            marginal_x = (len(X[feature_indices]) / len(Y)) 
            marginal_y = (len(cur_y) / len(Y))
            log_probability = math.log(joint_probability / (marginal_x * marginal_y))
            sum = sum + joint_probability * log_probability
    return sum

def mrmr(X, Y, n_features):
    chosen = []
    
    for _ in range(n_features):
        remaining_indices = []
        for index in range(X.shape[1]):
            if index not in chosen:
                remaining_indices.append(index)
        max_relevance = -100
        max_index = -1
        
        for index in remaining_indices:
            cur_column = X[:, index]
            relevance = find_mutual(X, index, Y)
            redundancy = 0
            
            for item in chosen:
                column = X[:, index]
                redundancy = redundancy + find_mutual(X, index, column)
                
            if len(chosen) != 0:
                average_redundancy = redundancy / len(chosen)
            else:
                average_redundancy = 0
            
            mrmr_val = relevance - average_redundancy
            if mrmr_val > max_relevance:
                max_relevance = mrmr_val 
                max_index = index
            
        chosen.append(max_index)
    return chosen
    




