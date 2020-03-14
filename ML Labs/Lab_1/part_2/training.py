from time import time
import numpy as np
from common.file_processor_tic_tac_toe import *
from plot_builder import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.naive_bayes import GaussianNB
import scikitplot as skplt
import matplotlib.pyplot as plt

# Create data
sum_data_count = 100
mu, sigma_1, sigma_2 = 0, 0.0002, 20
f1 = np.random.normal(mu, sigma_1, sum_data_count // 2)
f1 = np.reshape(f1, (-1, 1))
l1 = np.full(sum_data_count // 2, 0)
l1 = np.reshape(l1, (-1, 1))

f2 = np.random.normal(mu, sigma_2, sum_data_count // 2)
f2 = np.reshape(f2, (-1, 1))
l2 = np.full(sum_data_count // 2, 1)
l2 = np.reshape(l2, (-1, 1))

features = np.concatenate((f1, f2))
labels = np.concatenate((l1, l2))

# Setup
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.5, random_state=42, shuffle=True)
gnb = GaussianNB()

# Train
gnb.fit(features_train, labels_train)

# Predictions
labels_gnb_score = gnb.predict_proba(features_test)
labels_pred = gnb.predict(features_test)

# Analyse
skplt.metrics.plot_roc_curve(labels_test, labels_gnb_score)
skplt.metrics.plot_precision_recall(labels_test, labels_gnb_score)
skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True)
plt.show()

