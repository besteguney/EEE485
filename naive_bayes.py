import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class NaiveBayes:
    def __init__(self):
        self.classes = [0,1,2]
        self.mean = []
        self.var = []
        self.priors = []
        
    def fit(self, xtrain, ytrain):
        sample_size, feature_size = xtrain.shape
        n_classes = 3

        # calculating mean, var, prior 
        self.mean = np.zeros((n_classes, feature_size), dtype=np.float64)
        self.var = np.zeros((n_classes, feature_size), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64) # prior for each class
        
        for index in range(n_classes):
            indices = np.where(ytrain[:, 0] == index)[0]
            x_class = xtrain[indices]
            self.mean[index, :] = x_class.mean(axis=0)
            self.var[index, :] = x_class.var(axis=0)
            self.priors[index] = x_class.shape[0] / float(sample_size) 

    def predict(self, xtest):
        ypredict = np.zeros((xtest.shape[0], 1))
        cur = 0

        for x in xtest:
            posteriors = [] # posteriors for point x
            for index in range(len(self.classes)):
                prior = np.log(self.priors[index])
                posterior = np.sum(np.log(self.likelihood(index, x)))
                posterior = posterior + prior
                posteriors.append(posterior)
            ypredict[cur] = self.classes[np.argmax(posteriors)]
            cur = cur + 1
        return np.array(ypredict)
    
    ## Likelihood is gaussian
    def likelihood(self, class_index, xsample):
        mean = self.mean[class_index]
        var = self.var[class_index]

        return (np.exp(- ((xsample - mean) ** 2) / (2 * var))) / (np.sqrt(2 * np.pi * var))

    def error(self, ypredict, ytest):
        score = 0
        for index in range(ypredict.shape[0]):
            if ypredict[index][0] != ytest[index][0]:
                score = score + 1
        return score / ypredict.shape[0]

    def accuracy(self, ypredict, ytest):
        score = 0
        for index in range(ypredict.shape[0]):
            if ypredict[index][0] == ytest[index][0]:
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
                
            error_score = self.error(ypredict, ytest)
            scores = scores + error_score
            start_row = start_row + fold_size
            current_fold = current_fold - 1
        return scores / n_fold  