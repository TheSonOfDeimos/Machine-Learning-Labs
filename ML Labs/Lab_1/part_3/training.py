from time import time
import numpy
from common.file_processor_tic_tac_toe import *
from common.plot_builder import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
import scikitplot as skplt
import matplotlib.pyplot as plt_svm
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.datasets import load_energy

# SETUP
features, labels = return_features_labels()
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, train_size=0.5, random_state=42, shuffle=True)

plot_builder = PlotBuilder()

#for k in range(1, 20):
    # TRAIN
k = 2
kNN = KNeighborsClassifier(n_neighbors=k)
kNN.fit(features_train, labels_train)

# TEST
#predict_me = np.array([1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1])
#predict_me = predict_me.reshape(1, -1)

labels_gnb_score = kNN.predict_proba(features_test)
labels_pred = kNN.predict(predict_me)
print("Acc = ", kNN.score(features_test, labels_test))

# ANALYSE RESULTS
plot_builder.appendData(k, precision_score(
    labels_test, labels_pred, average='weighted'), recall_score(labels_test, labels_pred, average='weighted'), f1_score(labels_test, labels_pred, average='weighted'))


plot_builder.show()
skplt.metrics.plot_roc_curve(labels_test, labels_gnb_score)
skplt.metrics.plot_precision_recall(labels_test, labels_gnb_score)
plt_svm.show()

