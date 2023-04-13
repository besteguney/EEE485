import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class LogisticRegression:
    def __init__(self, lr, iteration):
        self.learning_rate = lr
        self.iteration = iteration

    def sigmoid(self, function):
        return 1.0 / (1 + np.exp(-function))

    def classify(self, prediction1, prediction2, prediction3):
        result = np.zeros((prediction1.shape[0], 3))
        for index in range(prediction1.shape[0]):
            probabilities = np.array([prediction1[index], prediction2[index], prediction3[index]])
        
            maximum = np.argmax(probabilities)
        
            one_hot = np.zeros(3)
            one_hot[maximum] = 1
            result[index] = one_hot
        return result
    
    def cost(self, features, labels, weight):
        # finding XB
        val = features @ weight
        prediction = self.sigmoid(val)
        sample_size = features.shape[0]
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
        score = 0
        for index in range(ypredict.shape[0]):
            equal = np.array_equal(ypredict[index], ytest[index])
            if not equal:
                score = score + 1
        return score / ypredict.shape[0]

    def k_fold_cross(self, df: pd.DataFrame, n_fold=10):
        df = df.sample(frac=1)
        fold_size = int(df.shape[0] / n_fold)
        start_row = 0
        scores = 0
        current_fold = n_fold

        # Converting data frame to numpy array
        x_matrix = df.iloc[:, :-3].values
        x_matrix = x_matrix.astype(float)
        y_matrix = df.iloc[:, -3:].values
        y1_vector = df.iloc[:, -3:-2].values
        y2_vector = df.iloc[:, -2:-1].values
        y3_vector = df.iloc[:, -1:].values

        while current_fold > 0:
            xtest = x_matrix[start_row:start_row + fold_size]
            ytest = y_matrix[start_row:start_row + fold_size]

            train1 = x_matrix[0:start_row]
            train2 = x_matrix[start_row + fold_size:]
            xtrain = np.concatenate((train1, train2), axis=0)


            ytrain_1 = np.concatenate((y1_vector[0:start_row], y1_vector[start_row + fold_size:]), axis=0)
            ytrain_2 = np.concatenate((y2_vector[0:start_row], y2_vector[start_row + fold_size:]), axis=0)
            ytrain_3 = np.concatenate((y3_vector[0:start_row], y3_vector[start_row + fold_size:]), axis=0)

            weight = np.zeros((13,1))
            param1 = self.fit(xtrain, ytrain_1, weight)
            param2 = self.fit(xtrain, ytrain_2, weight)
            param3 = self.fit(xtrain, ytrain_3, weight)
            
            predict1 = self.predict(xtest, param1)
            predict2 = self.predict(xtest, param2)
            predict3 = self.predict(xtest, param3)

            ypredict = self.classify(predict1, predict2, predict3)
                
            mse_score = self.mse(ypredict, ytest)
            print(mse_score)
            scores = scores + mse_score
            start_row = start_row + fold_size
            current_fold = current_fold - 1
        return scores / n_fold  