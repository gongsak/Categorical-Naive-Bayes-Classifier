import numpy as np
import pandas as pd
import math as m

with open('mushroom/Editedagaricus-lepiota.data') as file: #https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset 
    df = pd.read_csv(file)
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.columns)
    print(df.shape)

class NaiveBayesClassifier:
    def fit(self, X, y):



    def compute_likelihood(self, X):
        likelihood = 1;

        return likelihood


    
    def predict(self, X):
        predictions = []
        


