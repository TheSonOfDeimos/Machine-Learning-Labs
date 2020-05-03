import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer


def return_features_labels(file_path):
    data = pd.read_csv(file_path, sep=",")
    features = data.values
    features = (Imputer().fit_transform(features))
    features = features.astype(np.int)
    return features



