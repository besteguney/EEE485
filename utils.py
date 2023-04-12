import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Utils:

    def __init__(self, df):
        self.data = df
        self.features = df.iloc[:, :-1]
        self.response = df.loc[:, ['Adaptivity Level']]
        self.mapping = []

    def labeling(self):
        # Data Labeling
        columns = self.features.columns.tolist()
        labeled_df = self.features

        for column in columns:
            category = np.arange(0, self.data[column].nunique(), 1)
            unique = self.data[column].unique()
            mapping = {}
            for val, index in enumerate(category):
                mapping[unique[index]] = val
                self.mapping.append({column : mapping})
                #labeled_df[column] = self.data[column].astype('category')
                labeled_df[column] = self.data[column].map(mapping)
        return labeled_df

    def one_hot_encoding(self):
        one_hot_df = pd.get_dummies(self.response, columns=['Adaptivity Level'])
        return one_hot_df

def mse(ypredict, ytest):
    score = 0
    for index in range(ypredict.shape[0]):
        equal = np.array_equal(ypredict[index], ytest.iloc[index].values)
        if not equal:
            score = score + 1
    return score / ypredict.shape[0]

def k_fold_cross(model, df: pd.DataFrame, n_fold=10, logistic_regression=0):
    df = df.sample(frac=1)
    fold_size = int(df.shape[0] / n_fold)
    start_row = 0
    scores = 0
    current_fold = n_fold
    while current_fold > 0:
        test = df.iloc[start_row:start_row + fold_size,:]
        xtest = test.iloc[:, :-3]
        ytest = test.iloc[:, -3:]

        train1 = df.iloc[0:start_row]
        train2 = df.iloc[start_row + fold_size:]
        train = train1.append(train2, ignore_index=True)

        xtrain = train.iloc[:, :-3]

        if logistic_regression == 1:
            ytrain_1 = train.iloc[:, -1:]
            ytrain_2 = train.iloc[:, -2:-1]
            ytrain_3 = train.iloc[:, -3:-2]

            weight = np.zeros((13,1))
            param1 = model.fit(xtrain, ytrain_1, weight)
            param2 = model.fit(xtrain, ytrain_2, weight)
            param3 = model.fit(xtrain, ytrain_3, weight)
        
            predict1 = model.predict(xtest, param1)
            predict2 = model.predict(xtest, param2)
            predict3 = model.predict(xtest, param3)

            ypredict = model.classify(predict1, predict2, predict3)
            
        mse_score = mse(ypredict, ytest)
        scores = scores + mse_score
        start_row = start_row + fold_size
        current_fold = current_fold - 1
    return scores / n_fold   
