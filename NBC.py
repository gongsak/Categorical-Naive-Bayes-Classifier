import numpy as np
import pandas as pd
import math as m

with open('mushroom/Editedagaricus-lepiota.data') as file:
    df = pd.read_csv(file)
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.columns)
    print(df.shape)

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
