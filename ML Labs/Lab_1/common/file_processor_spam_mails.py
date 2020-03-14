import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer

data = pd.read_csv("./Lab 1/lab1_data/spam.csv", sep=",")
data_copy = pd.read_csv("./Lab 1/lab1_data/spam.csv", sep=",")  # for further use


def return_features_labels():
    global data
    global data_copy

    # Positive is win, negative is lose
    mapping_for_type = {"spam": 1, "nonspam": 0}
    data.type = data.type.map(mapping_for_type)
    data_copy.type = data_copy.type.map(mapping_for_type)

    data = data.drop(columns=["type"], axis=1)
    data = data.drop(columns=["num"], axis=1)

    # Extracting features and labels
    features = data.values
    labels = data_copy.type.values

    # Filling missing values aka "b" with the mean
    #features = (Imputer().fit_transform(features))

    #features = features.astype(np.int)
    #labels = labels.astype(np.int)

    return features, labels
