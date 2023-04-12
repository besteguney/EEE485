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
    
       #one_hot_df = pd.get_dummies(self.data, columns=['Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student',
       #'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
       #'Network Type', 'Class Duration', 'Self Lms', 'Device','Adaptivity Level'])
       # return one_hot_df

    def normalize(self):
        mean = np.mean(self.data, axis=0)
        print(mean)
        return
    
        