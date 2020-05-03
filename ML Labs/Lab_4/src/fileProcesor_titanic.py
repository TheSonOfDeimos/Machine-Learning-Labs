import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as Imputer


def stringToInt(str_in):
    if (isinstance(str_in, int) or isinstance(str_in, float)):
        return str_in
    new_str = ""
    for i in range(len(str_in)):
        new_str += str(ord(str_in[i]))
    return new_str

def ifnan(str_in):
    if str_in is np.nan:
        return 0
    return str_in

def replace_non_numeric(df):
    df["Sex"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
    df["Ticket"] = df["Ticket"].apply(lambda ticket: stringToInt(ticket))
    df["Cabin"] = df["Cabin"].apply(lambda ticket: stringToInt(ticket))
    df["Embarked"] = df["Embarked"].apply(lambda ticket: stringToInt(ticket))
    df["Embarked"] = df["Embarked"].apply(lambda ticket: stringToInt(ticket))
    df["Embarked"] = df["Embarked"].apply(lambda ticket: ifnan(ticket))
    return df

filecount = 0
def return_features_labels(file_path):
    global filecount
    filecount += 1
    data = replace_non_numeric(pd.read_csv(file_path, sep=","))
    data_copy = replace_non_numeric(pd.read_csv(file_path, sep=","))

    print(data.head)
    data = data.drop(columns=["Embarked"], axis=1)
    data = data.drop(columns=["PassengerId"], axis=1)
    data = data.drop(columns=["Name"], axis=1)


   
    # Extracting features and labels
    features = data.values
    labels = data_copy.Embarked.values

    features = (Imputer().fit_transform(features))
    features = features.astype(np.int)
    labels = labels.astype(np.int)



    return features, labels
