import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)

def nearest_neighbour(n_neighbors, X):
    distances = np.zeros((X.shape[0], X.shape[0]))
    neigbor_indices = np.zeros((X.shape[0], n_neighbors))
    for index in range(X.shape[0]):
        distances[index] = [euclidean_distance(X[index], sample) for sample in X]
        neigbor_indices[index] = np.argsort(distances[index])[:n_neighbors]
    return neigbor_indices

class Smote:
    def __init__(self, df: pd.DataFrame, amount, n_neighbors):
        self.data = df
        self.minority_samples = self.minority_samples()
        self.amount = amount
        self.n_neighbors = n_neighbors

    def minority_samples(self):
        return self.data[self.data['Adaptivity Level'] == 2]

    def smote(self):
        features = self.minority_samples.drop(['Adaptivity Level'], axis=1)
        features_numpy = features.values
        sample_size = self.minority_samples.shape[0]
        synthetic_data = np.zeros((int(self.amount/100)*sample_size, features.shape[1]))
        amount = int(self.amount/100)
        indices = nearest_neighbour(5, features_numpy)
        # populating
        synthetic_index = 0
        
        for index, cur_indices in enumerate(indices):
            cur_amount = amount
            while cur_amount > 0:
                random_index = random.randint(0, indices.shape[1] - 1)
                difference = features.iloc[int(cur_indices[random_index])] - features.iloc[index]
                gap = random.uniform(0,1)
                synthetic_data[synthetic_index, :] = features.iloc[index] + gap * difference
                synthetic_index = synthetic_index + 1
                cur_amount = cur_amount - 1

        synthetic_data = (np.rint(synthetic_data)).astype(int)
        return synthetic_data