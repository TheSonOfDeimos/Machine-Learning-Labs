import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer

data = pd.read_csv(
    "./Lab_1/lab1_data/spam7.csv", sep=",")
data_copy = pd.read_csv(
    "./Lab_1/lab1_data/spam7.csv", sep=",")


def return_features_labels():
    global data
    global data_copy

    data = data.drop(columns=["yesno"], axis=1)

    # Extracting features and labels
    features = data.values
    labels = data_copy.yesno.values



    return features, labels
