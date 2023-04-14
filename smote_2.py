# Handling Imbalances in Target Value - SMOTE
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Smote:
    def __init__(self, df: pd.DataFrame, amount, n_neighbors):
        self.data = df
        self.minority_samples = self.minority_samples()
        self.amount = amount
        self.n_neighbors = n_neighbors

    def nearest_neighbour(self, X):
        nbs = NearestNeighbors(n_neighbors=self.n_neighbors,metric='euclidean',algorithm='kd_tree').fit(X)
        euclidean,indices= nbs.kneighbors(X)
        return indices

    def minority_samples(self):
        return self.data[self.data['Adaptivity Level'] == 2]

    def smote(self):
        features = self.minority_samples.drop(['Adaptivity Level'], axis=1)
        sample_size = self.minority_samples.shape[0]
        synthetic_data = np.zeros((int(self.amount/100)*sample_size, features.shape[1]))
        amount = int(self.amount/100)
        indices = self.nearest_neighbour(features)

        # populating
        synthetic_index = 0
        
        for index, cur_indices in enumerate(indices):
            cur_amount = amount
            while cur_amount > 0:
                random_index = random.randint(0, indices.shape[1] - 1)
                difference = features.iloc[cur_indices[random_index]] - features.iloc[index]
                gap = random.uniform(0,1)
                synthetic_data[synthetic_index, :] = features.iloc[index] + gap * difference
                synthetic_index = synthetic_index + 1
                cur_amount = cur_amount - 1

        synthetic_data = (np.rint(synthetic_data)).astype(int)
        return synthetic_data