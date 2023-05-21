import pandas as pd
import numpy as np
from decision_trees import Tree

class RandomForest():
    def __init__(self, max_depth=10, n_features=10, n_trees=10, mode=1):
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_trees = n_trees
        self.mode = mode
    
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
            decision_tree = Tree(max_depth=self.max_depth, n_features=self.n_features, mode=self.mode)

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

    def accuracy(self, ypredict, ytest):
        score = 0
        for index in range(len(ypredict)):
            if ytest[index] == ypredict[index]:
                score = score + 1
        return score / len(ypredict)

    def error(self, ypredict, ytest):
        score = 0
        for index in range(len(ypredict)):
            if ytest[index] != ypredict[index]:
                score = score + 1
        return score / len(ypredict)
    
    def k_fold_cross(self, df: pd.DataFrame, n_fold=10):
        df = df.sample(frac=1)
        fold_size = int(df.shape[0] / n_fold)
        start_row = 0
        scores = 0
        current_fold = n_fold

        # Converting data frame to numpy array
        x_matrix = df.iloc[:, :-1].values
        x_matrix = x_matrix.astype(int)
        y_vector = df.iloc[:, -1:].values
        y_vector = y_vector.astype(int)

        while current_fold > 0:
            xtest = x_matrix[start_row:start_row + fold_size]
            ytest = y_vector[start_row:start_row + fold_size]

            train1 = x_matrix[0:start_row]
            train2 = x_matrix[start_row + fold_size:]
            xtrain = np.concatenate((train1, train2), axis=0)
            ytrain = np.concatenate((y_vector[0:start_row], y_vector[start_row + fold_size:]), axis=0)
            
            self.fit(xtrain, ytrain)
            ypredict = self.predict(xtest)
                
            error_score = self.error(ypredict, ytest)
            scores = scores + error_score
            start_row = start_row + fold_size
            current_fold = current_fold - 1
        return scores / n_fold  

