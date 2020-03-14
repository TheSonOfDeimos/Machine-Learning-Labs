import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer




def return_features_labels(file_path):
    data = pd.read_csv(file_path, sep='\t', skiprows=1)
    data_copy = pd.read_csv(file_path, sep='\t', skiprows=1)

    data.columns = ["Id", "X1", "X2", "Color"]
    data_copy.columns = ["Id", "X1", "X2", "Color"]

    mapping_for_color = {"red": 1, "green": 0}
    data.Color = data.Color.map(mapping_for_color)
    data_copy.Color = data_copy.Color.map(mapping_for_color)

    data = data.drop(columns=["Id"], axis=1)
    data = data.drop(columns=["Color"], axis=1)

    # Extracting features and labels
    features = data.values
    labels = data_copy.Color.values

    return features, labels
