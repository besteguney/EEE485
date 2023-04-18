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

def mse(ypredict, ytest):
    score = 0
    for index in range(ypredict.shape[0]):
        if ypredict[index] != ytest[index][0]:
            score = score + 1
    return score / ypredict.shape[0]

def k_fold_cross(model, df: pd.DataFrame, n_fold=10):
    df = df.sample(frac=1)
    fold_size = int(df.shape[0] / n_fold)
    start_row = 0
    scores = 0
    current_fold = n_fold

    # Converting data frame to numpy array
    x_matrix = df.iloc[:, :-1].values
    x_matrix = x_matrix.astype(float)
    y_vector = df.iloc[:, -1:].values

    while current_fold > 0:
        xtest = x_matrix[start_row:start_row + fold_size]
        ytest = y_vector[start_row:start_row + fold_size]

        train1 = x_matrix[0:start_row]
        train2 = x_matrix[start_row + fold_size:]
        xtrain = np.concatenate((train1, train2), axis=0)
        ytrain = np.concatenate((y_vector[0:start_row], y_vector[start_row + fold_size:]), axis=0)

        model.fit(xtrain, ytrain)
        ypredict = model.predict(xtest)
                
        mse_score = mse(ypredict, ytest)
        scores = scores + mse_score
        start_row = start_row + fold_size
        current_fold = current_fold - 1
    return scores / n_fold 

def test_train_split(percentage, X, Y):
    sample_size = int(X.shape[0] / 100) * percentage
    test_indices = np.random.randint(X.shape[0], size=sample_size)
    remaining_indices = np.delete(np.arange(X.shape[0]), test_indices)

    xtest = X[test_indices]
    ytest = Y[test_indices]

    xtrain = X[remaining_indices]
    ytrain = Y[remaining_indices]
    return xtest, ytest, xtrain, ytrain, test_indices, remaining_indices

def confusion_matrix(ypredict, ytest):
    tp = [0,0,0] # moderate, low, high
    fp = [0,0,0]
    fn = [0,0,0]
    tn = [0,0,0]
    
    for index in range(ypredict.shape[0]):
        if ypredict[index] == ytest[index]:
            val = ypredict[index]
            tp[val] = tp[val] + 1
        else:
            if ypredict[index] == 0:
                if ytest[index] == 1:
                    fn[1] = fn[1] + 1
                    tn[2] = tn[2] + 1
                else:
                    fn[2] = fn[2] + 1
                    tn[1] = tn[1] + 1
                fp[0] = fp[0] + 1
            elif ypredict[index] == 1:
                if ytest[index] == 0:
                    fn[0] = fn[0] + 1
                    tn[2] = tn[2] + 1
                else:
                    fn[2] = fn[2] + 1
                    tn[0] = tn[0] + 1
                fp[1] = fp[1] + 1
            else:
                if ytest[index] == 1:
                    fn[1] = fn[1] + 1
                    tn[0] = tn[0] + 1
                else:
                    fn[0] = fn[0] + 1
                    tn[1] = tn[1] + 1
                fp[2] = fp[2] + 1

    precision = [0,0,0]
    recall = [0,0,0]
    f1 = [0,0,0]
    for index in range(3):
        precision[index] = tp[index] / tp[index] + fp[index]
        recall[index] = tp[index] / tp[index] + tn[index]
        f1[index] = 2 * (precision[index] * recall[index]) / (precision[index] + recall[index])
    return np.array([precision, recall, f1]).T