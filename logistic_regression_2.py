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
            probabilities = np.array([prediction1.iloc[index], prediction2.iloc[index], prediction3.iloc[index]])
        
            maximum = np.argmax(probabilities)
        
            one_hot = np.zeros(3)
            one_hot[maximum] = 1
            result[index] = one_hot
        return result
    
    def cost(self, features, labels, weight):
        # finding XB
        features = features.astype(float)
        val = features @ weight
        prediction = self.sigmoid(val)
        sample_size = features.shape[0]
        prediction.columns = [labels.columns[0]]
        return (features.T @ (prediction - labels)) / sample_size

    def gradient_descent(self, features, labels, weight):
        new_weight = weight - self.learning_rate * self.cost(features, labels, weight)
        return new_weight

    def fit(self, xtrain, ytrain, weight):
        for index in range(self.iteration):
            weight = self.gradient_descent(xtrain, ytrain, weight)
        return weight
    
    def predict(self, features, weight):
        features = features.astype(float)
        prediction = features @ weight
        self.sigmoid(prediction)
        return prediction