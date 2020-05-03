import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer

data = pd.read_csv(
    "data/vehicle.csv", sep=",")
data_copy = pd.read_csv(
    "data/vehicle.csv", sep=",")


def return_features_labels():
    global data
    global data_copy

    data = data.drop(columns=["Class"], axis=1)

    # Extracting features and labels
    features = data.values
    labels = data_copy.Class.values

    return features, labels
