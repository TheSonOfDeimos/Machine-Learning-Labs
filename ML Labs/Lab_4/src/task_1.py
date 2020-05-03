from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from fileProcesor_glass import *
from plot_builder import *



# Prepare dataframes
features, labels = return_features_labels()
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.8, random_state=42, shuffle=True)

# Init estimators
base_cls = DecisionTreeClassifier()
base_knn= KNeighborsClassifier()
base_svm = svm.SVC(kernel='linear')

for estimator in (base_cls, base_knn, base_svm):
    # Init plotbuilder
    plot_builder = PlotBuilder()
    for n_estimators in (1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048):
        model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators, random_state=8)
        model.fit(features_train, labels_train)
        labels_pred = model.predict(features_test)

        plot_builder.appendData(n_estimators, precision_score(labels_test, labels_pred, average='weighted'),
                                recall_score(labels_test, labels_pred, average='weighted'),
                                f1_score(labels_test, labels_pred, average='weighted'))
    plot_builder.show("Model = " + estimator.__class__.__name__)
