import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from smote import euclidean_distance

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def predict(self, xtest):
        ypredict = [self.predict_sample(x) for x in xtest]
        return np.array(ypredict)

    def predict_sample(self, x):
        distances = [euclidean_distance(x, sample) for sample in self.xtrain]
        neigbor_indices = np.argsort(distances)[:self.n_neighbors]
        neigbor_labels = [self.ytrain[index] for index in neigbor_indices]
        neigbor_count = [0,0,0]
        for val in neigbor_labels:
            val = int(val)
            neigbor_count[val] = neigbor_count[val] + 1
        return np.argmax(neigbor_count)
    
    def accuracy(self, ypredict, ytest):
        score = 0
        for index in range(ypredict.shape[0]):
            if ypredict[index] == ytest[index]:
                score = score + 1
        return score / ypredict.shape[0]
    
    def min_max_scaling(self, xtrain):
        X = xtrain.copy()

        for index in range(X.shape[1]):
            min_value = np.min(X[:, index])
            max_value = np.max(X[:, index])
            X[:, index] = (X[:, index] - min_value ) / (max_value) - min_value
        return X

    def error(self, ypredict, ytest):
        score = 0
        for index in range(ypredict.shape[0]):
            if ypredict[index] != ytest[index][0]:
                score = score + 1
        return score / ypredict.shape[0]

    def k_fold_cross(self, df: pd.DataFrame, n_fold=10):
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

            self.fit(xtrain, ytrain)
            ypredict = self.predict(xtest)
                    
            error_score = error(ypredict, ytest)
            scores = scores + error_score
            start_row = start_row + fold_size
            current_fold = current_fold - 1
        return scores / n_fold 
