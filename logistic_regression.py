import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class LogisticRegression:
    def __init__(self, lr, iteration, is_regularized=False, regularization_type=0, tune_param=0):
        self.learning_rate = lr
        self.iteration = iteration
        self.is_regularized = is_regularized
        self.tune_param = tune_param
        self.regular_type = regularization_type

    def sigmoid(self, function):
        return 1.0 / (1 + np.exp(-function))

    def classify(self, prediction1, prediction2, prediction3):
        result = np.zeros((prediction1.shape[0],1))
        for index in range(prediction1.shape[0]):
            probabilities = np.array([ prediction1[index], prediction2[index],prediction3[index]])       
            maximum = np.argmax(probabilities)
            result[index] = 2-maximum
        return result
    
    def cost(self, features, labels, weight):
        # finding XB
        val = features @ weight
        prediction = self.sigmoid(val)
        sample_size = features.shape[0]

        if self.is_regularized:
            if self.regular_type == 2:
                return ((features.T @ (prediction - labels)) + (self.tune_param * (weight.T @ weight) )) / sample_size
            else:
                return ((features.T @ (prediction - labels)) + (self.tune_param * np.sum(np.abs(weight))) ) / sample_size
        return (features.T @ (prediction - labels)) / sample_size

    def gradient_descent(self, features, labels, weight):
        new_weight = weight - self.learning_rate * self.cost(features, labels, weight)
        return new_weight

    def fit(self, xtrain, ytrain, weight):
        for index in range(self.iteration):
            weight = self.gradient_descent(xtrain, ytrain, weight)
        return weight
    
    def predict(self, features, weight):
        prediction = features @ weight
        self.sigmoid(prediction)
        return prediction

    def mse(self, ypredict, ytest):
        return np.sum(ypredict != ytest) / ypredict.shape[0]

    def k_fold_cross(self, df: pd.DataFrame, weight, n_fold=10):
        df = df.sample(frac=1)
        fold_size = int(df.shape[0] / n_fold)
        start_row = 0
        scores = 0
        current_fold = n_fold

        # Converting data frame to numpy array
        x_matrix = df.iloc[:, :-3].values
        x_matrix = x_matrix.astype(float)
        
        y1_vector = df.iloc[:, -3:-2].values # high
        y2_vector = df.iloc[:, -2:-1].values # low
        y3_vector = df.iloc[:, -1:].values # moderate

        y_vector = np.zeros((y1_vector.shape[0], 1))
        for index in range(y1_vector.shape[0]):
            vals = np.array([ y1_vector[index], y2_vector[index], y3_vector[index]])       
            maximum = np.argmax(vals)
            y_vector[index] = 2 - maximum

        while current_fold > 0:
            xtest = x_matrix[start_row:start_row + fold_size]
            ytest = y_vector[start_row:start_row + fold_size]

            train1 = x_matrix[0:start_row]
            train2 = x_matrix[start_row + fold_size:]
            xtrain = np.concatenate((train1, train2), axis=0)

            ytrain_1 = np.concatenate((y1_vector[0:start_row], y1_vector[start_row + fold_size:]), axis=0)
            ytrain_2 = np.concatenate((y2_vector[0:start_row], y2_vector[start_row + fold_size:]), axis=0)
            ytrain_3 = np.concatenate((y3_vector[0:start_row], y3_vector[start_row + fold_size:]), axis=0)

            param1 = self.fit(xtrain, ytrain_1, weight)
            param2 = self.fit(xtrain, ytrain_2, weight)
            param3 = self.fit(xtrain, ytrain_3, weight)
            
            predict1 = self.predict(xtest, param1)
            predict2 = self.predict(xtest, param2)
            predict3 = self.predict(xtest, param3)

            ypredict = self.classify(predict1, predict2, predict3)
                
            mse_score = self.mse(ypredict, ytest)
            scores = scores + mse_score
            start_row = start_row + fold_size
            current_fold = current_fold - 1
        return scores / n_fold  