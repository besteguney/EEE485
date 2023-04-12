import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class LogisticRegression:
    def __init__(self, train: pd.DataFrame, lr, iteration):
        self.data = train
        self.learning_rate = lr
        self.iteration = iteration
        self.features = train.iloc[:, :-1].astype(float)
        self.labels = train.iloc[:, -1:]
        #self.labels = self.labels.set_index(self.labels.columns[0])
        self.weight = np.zeros((13,1))

    def sigmoid(self, function):
        return 1.0 / (1 + np.exp(-function))

    def classify(self, prediction1, prediction2, prediction3, boundary=0.50):
        return prediction > boundary
    
    def rss(self):
        # finding XB
        val = self.features @ self.weight
        prediction = self.sigmoid(val)
        sample_size = len(self.features)
        prediction.columns = [self.labels.columns[0]]
        return (self.features.T @ (prediction - self.labels)) / sample_size

    def gradient_descent(self):
        self.weight = self.weight - self.learning_rate * self.rss()
        return

    def prediction(self, features):
        for index in range(self.iteration):
            self.gradient_descent()
        prediction = features @ self.weight
        self.sigmoid(prediction)
        #prediction = self.classify(prediction)
        return prediction