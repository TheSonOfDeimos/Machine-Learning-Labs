import numpy
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    brier_score_loss, precision_score, recall_score, f1_score)
import scikitplot as skplt
from sklearn import svm
import matplotlib.pyplot as plt

from common.plot_builder import *
from common.file_processor_svmdata import *

# SETUP
features_train, labels_train = return_features_labels(
    "./Lab_1/lab1_data/svmdata_b.txt")
features_test, labels_test = return_features_labels(
    "./Lab_1/lab1_data/svmdata_b_test.txt")

plot_builder = PlotBuilder()

for c in range(1, 20):
        # TRAIN
        lsvc = svm.SVC(kernel='linear', C=c)
        lsvc.fit(features_train, labels_train)

        # TEST
        labels_pred = lsvc.predict(features_test)

        print("Accuracy on Train with C = ", c, " and score = ", lsvc.score(features_train, labels_train))
        print("Accuracy on Test with C = ", c, " and score = ", lsvc.score(features_test, labels_test))
        print("\n")

# ANALYSE RESULTS
fig, sub = plt.subplots(1, 2)
ax = sub.flatten()
plot_builder.plot_svm(ax[0], lsvc, features_test, labels_test)
skplt.metrics.plot_confusion_matrix(
labels_test, labels_pred, normalize=True, ax=ax[1])
plt.show()




