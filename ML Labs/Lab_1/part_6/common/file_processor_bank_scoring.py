import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer

def return_features_labels(file_path):
    data = pd.read_csv(file_path, sep='\t')
    data_copy = pd.read_csv(file_path, sep='\t')

    data = data.drop(columns=["SeriousDlqin2yrs"], axis=1)

    # Extracting features and labels
    features = data.values
    labels = data_copy.SeriousDlqin2yrs.values



    return features, labels
