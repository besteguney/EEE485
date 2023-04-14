import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Class for the methods of PCA process. This applied to reduce the dimension of the original data.

# 2. PCA
class PCA:
    
    def __init__(self, features):
        self.features = features
        self.eigen_vals, self.eigen_vecs = self.eigens(self.features)
        
    def normalize(self, features):
        centered_features = features - np.mean(features, axis=0)
        return centered_features
    
    def eigens(self, features):
        covariance_matrix = np.cov(self.features, rowvar=False)
        eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
        return eigen_vals, eigen_vecs
    
    def examine(self, component_count):
        sorted_index = np.argsort(self.eigen_vals)[::-1]
        sorted_eigen_val = self.eigen_vals[sorted_index]
        sorted_eigen_vec = self.eigen_vecs[:,sorted_index]
        print(sorted_eigen_vec.shape)
        eigenvector_subset = sorted_eigen_vec[:,0:component_count]
        return eigenvector_subset

    def pve(self, n_component):
        # variance captured by first n component
        pcs = self.examine(n_component)

        var_all = 0
        for j in range(13):
            for i in range(len(self.features)):
                var_all = var_all + (self.features[i][j] ** 2)

        var_captured = 0
        for i in range(len(self.features)):   
            for j in range(n_component):
                var_captured = var_captured + (self.features[i] @ pcs[:,j]) ** 2
        
        return np.sum(var_captured/var_all)



