import pandas as pd
import numpy as np
from decision_trees import Tree

class RandomForest():
    def __init__(self, max_depth=10, min_split=2, min_inpurity=0.1, n_features=10, n_trees=10):
        self.max_depth = max_depth
        self.min_split = min_split
        self.min_inpurity = min_inpurity
        self.n_features = n_features
        self.n_trees = n_trees
    
    def predict(self, xtest):
        predictions = np.array([tree.predict(xtest) for tree in self.tree_list])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        actual_predict = np.zeros(len(tree_predictions))

        for index,tree_prediction in enumerate(tree_predictions):
            arr = [0,0,0]
            for val in tree_prediction:
                arr[val] = arr[val] + 1
            actual_predict[index] = np.argmax(arr)
        return actual_predict

    def fit(self, x_train, y_train):
        self.tree_list = []
        for index in range(self.n_trees):
            decision_tree = Tree(max_depth=self.max_depth, min_split=self.min_split, min_inpurity=self.min_inpurity, n_features=self.n_features)

            sampled_indeces = self.bagging(x_train, y_train)
            x_sampled = x_train[sampled_indeces]
            y_sampled = y_train[sampled_indeces]

            decision_tree.fit(x_sampled, y_sampled)
            self.tree_list.append(decision_tree)


    def bagging(self, X, y):
        n_samples = X.shape[0]
        # can select with replacement
        indeces = np.random.choice(n_samples, n_samples, replace=True)
        return indeces


