import pandas as pd
from sklearn import preprocessing

def getFeaturesLabels(filePath):
    data = pd.read_csv(filePath, sep = ",")
    print(data.head())

    # Extracting features and labels
    features = data.iloc[:, 0:2]
    print(features.head())
    labels = data.select_dtypes(include=[int])
    print(labels.head())

    le = preprocessing.LabelEncoder()
    labels = labels.apply(le.fit_transform)

    return features, labels
