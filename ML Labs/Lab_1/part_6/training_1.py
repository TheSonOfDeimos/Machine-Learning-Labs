import numpy
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from common.plot_builder import *
from common.file_processor_bank_scoring import *

# SETUP
features_train, labels_train = return_features_labels(
    "./Lab_1/lab1_data/bank_scoring_train.csv")
features_test, labels_test = return_features_labels(
    "./Lab_1/lab1_data/bank_scoring_test.csv")

#========================== kNN ========================================
# TRAIN
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(features_train, labels_train)

# TEST
labels_gnb_score = clf.predict_proba(features_test)
labels_pred = clf.predict(features_test)

# ANALYSE RESULTS
fig, sub = plt.subplots(2, 3)
ax = sub.flatten()
skplt.metrics.plot_roc_curve(labels_test, labels_gnb_score, title="ROC kNN", ax=ax[0])
skplt.metrics.plot_precision_recall(labels_test, labels_gnb_score, title="Recall kNN", ax=ax[1])
skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True, title="Confusion matrix kNN", ax=ax[2])

#========================== Tree ========================================
# TRAIN
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

# TEST
labels_gnb_score = clf.predict_proba(features_test)
labels_pred = clf.predict(features_test)

# ANALYSE RESULTS

skplt.metrics.plot_roc_curve(labels_test, labels_gnb_score, title="ROC Tree", ax=ax[3])
skplt.metrics.plot_precision_recall(labels_test, labels_gnb_score, title="Recall Tree", ax=ax[4])
skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True, title="Confusion matrix Tree", ax=ax[5])

plt.show()
