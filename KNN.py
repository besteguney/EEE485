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
    

