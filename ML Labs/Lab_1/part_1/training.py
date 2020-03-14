from time import time
import numpy
from common.file_processor_tic_tac_toe import *
from plot_builder import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.naive_bayes import GaussianNB

plot_builder = PlotBuilder()

features, labels = return_features_labels()
for train in numpy.arange(0.05, 1, 0.05):
        # Trinnig
        
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=train, random_state=42, shuffle=True)

        gnb = GaussianNB()
        gnb.fit(features_train, labels_train)

        # Predictions
        labels_pred = gnb.predict(features_test)

        plot_builder.appendData(train, precision_score(
            labels_test, labels_pred, average='weighted'), recall_score(labels_test, labels_pred, average='weighted'), f1_score(labels_test, labels_pred, average='weighted'))

plot_builder.show()

