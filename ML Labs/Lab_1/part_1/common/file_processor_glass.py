import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer

data = pd.read_csv(
    "./Lab_1/lab1_data/glass.csv", sep=",")
data_copy = pd.read_csv(
    "./Lab_1/lab1_data/glass.csv", sep=",")


def return_features_labels():
    global data
    global data_copy

    data = data.drop(columns=["Type"], axis=1)
    data = data.drop(columns=["Id"], axis=1)

    # Extracting features and labels
    features = data.values
    labels = data_copy.Type.values

    # Filling missing values aka "b" with the mean
    #features = (Imputer().fit_transform(features))

    #features = features.astype(np.int)
    #labels = labels.astype(np.int)

    return features, labels
