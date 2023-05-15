import pandas as pd
import numpy as np
from decision_node import TreeNode


class Tree():
    # Providing some stopping conditions
    def __init__(self, max_depth=10, min_split=2, n_features=None, root_node=None, mode=1):
        self.max_depth = max_depth
        self.min_split = min_split
        self.n_features = n_features
        self.root_node = root_node
        self.mode = mode 

    def predict(self, X):
        return np.array([self.predict_sample(xsample, self.root_node) for xsample in X])

    def predict_sample(self, xsample, node):
        if node.is_leaf():
            return node.val
        
        # BST logic
        if xsample[node.feature] <= node.threshold:
            return self.predict_sample(xsample, node.l_child)
        return self.predict_sample(xsample, node.r_child)

    
    def fit(self, x_train, y_train, cur_depth=0):
        self.n_features = x_train.shape[1] if not self.n_features else min(x_train.shape[1],self.n_features)
        self.root_node = self.enlarge(x_train, y_train, cur_depth)

    ## helper methods
    def entropy(self, labels):
        n_sample = len(labels)
        arr_label = [0, 0, 0]
        for label in labels:
            arr_label[label] = arr_label[label] + 1
        
        arr = np.array(arr_label)
        probabilities = arr / n_sample
        sum = 0
        for probability in probabilities:
            if probability > 0:
                sum = sum + (probability * np.log(probability))
        return -sum

    def gini_index(self, labels):
        n_sample = len(labels)
        arr_label = [0, 0, 0]
        for label in labels:
            arr_label[label] = arr_label[label] + 1
        
        arr = np.array(arr_label)
        probabilities = arr / n_sample
        sum = 0
        for probability in probabilities:
            if probability > 0:
                sum = sum + (probability * (1 - probability))
        return 1 - sum

    def calculate_impurity(self, feature, labels, threshold):
        node_impurity = self.gini_index(labels)
        l_child, r_child = self.split_node(feature, threshold)
        n_labels = len(labels)

        if len(l_child) == 0 or len(r_child) == 0:
            return 0
        left_impurity = self.gini_index(labels[l_child])
        right_impurity = self.gini_index(labels[r_child])
        child_impurity = (len(l_child) / n_labels) * left_impurity + (len(r_child) / n_labels) * right_impurity
        return node_impurity - child_impurity

    def calculate_gain(self, feature, labels, threshold):
        node_entropy = self.entropy(labels)
        l_child, r_child = self.split_node(feature, threshold)
        n_labels = len(labels)

        if len(l_child) == 0 or len(r_child) == 0:
            return 0

        left_entropy = self.entropy(labels[l_child])
        right_entropy = self.entropy(labels[r_child])
        child_entropy = (len(l_child) / n_labels) * left_entropy + (len(r_child) / n_labels) * right_entropy
        return node_entropy - child_entropy

    def enlarge(self, x_train, y_train, cur_depth):
        n_sample, n_features =  x_train.shape
        labels = np.unique(y_train)

        if (cur_depth >= self.max_depth or n_sample < self.min_split or len(labels) == 1) :
            # maximum value
            arr_label = [0, 0, 0]
            for val in y_train:
                arr_label[val] = arr_label[val] + 1
            leaf_val = np.argmax(arr_label)
            return TreeNode(l_child=None, r_child=None, val=leaf_val)

            
        # Randomly selecting features
        indices = np.random.choice(n_features, self.n_features, replace=False)
        best_thresholds, best_index = self.find_best_split(x_train, y_train, indices)

        best_feature = x_train[:, best_index]
        l_child, r_child = self.split_node(best_feature, best_thresholds)
        # recursively growing left and right indeces
        left_tree = self.enlarge(x_train[l_child, :], y_train[l_child], cur_depth+1)
        right_tree = self.enlarge(x_train[r_child, :], y_train[r_child], cur_depth+1)
        return TreeNode(left_tree, right_tree, best_index, best_thresholds)


    def find_best_split(self, X, Y, indices):
        if self.mode == 1:
            return self.find_best_split_entropy(X, Y, indices)
        else:
            return self.find_best_split_gini(X, Y, indices)

    def find_best_split_gini(self, X, Y, indices):
        min_impurity, min_index, min_threshold = 2, None, None

        for index in indices:
            cur_feature = X[:, index]
            values = np.unique(cur_feature)

            for val in values:
                cur_impurity = self.calculate_impurity(cur_feature, Y, val)
                if cur_impurity < min_impurity:
                    min_impurity = cur_impurity
                    min_index = index
                    min_threshold = val

        return min_threshold, min_index

    def find_best_split_entropy(self, X, Y, indices):
        max_gain, max_index, max_threshold = -1, None, None

        for index in indices:
            cur_feature = X[:, index]
            values = np.unique(cur_feature)

            for val in values:
                cur_gain = self.calculate_gain(cur_feature, Y, val)

                if cur_gain > max_gain:
                    max_gain = cur_gain
                    max_index = index
                    max_threshold = val

        return max_threshold, max_index

    def split_node(self, feature, threshold):
        l_child = np.argwhere(feature <= threshold).flatten()
        r_child = np.argwhere(feature > threshold).flatten()
        return l_child, r_child