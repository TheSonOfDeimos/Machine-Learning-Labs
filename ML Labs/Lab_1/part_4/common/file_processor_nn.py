import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer

data = pd.read_csv("./Lab_2/data/nn_1.csv", sep=",",usecols=["X1", "X2"])
data_copy = pd.read_csv("./Lab_2/data/nn_1.csv", sep=",", usecols=["class"])


def return_features_labels():
   #data = pd.read_csv("./Lab_2/data/nn.csv", sep=",")

   # Extracting features and labels
   features = data.values

   #mapping_for_wins = {1 : 1, -1 : 0}
   #data_copy.Class = data_copy.Class.map(mapping_for_wins)
   labels = data_copy.values
   
   
  

   return features, labels
