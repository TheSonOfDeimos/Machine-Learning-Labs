from subprocess import call
from IPython.display import Image
import numpy
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

from common.file_processor_spam7 import *
from subprocess import call
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz


# SETUP
features, labels = return_features_labels()
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, train_size=0.8, random_state=42, shuffle=True)

# TRAIN
clf = tree.DecisionTreeClassifier(splitter='best', max_depth=5)
classifier = clf.fit(features_train, labels_train)

# TEST
labels_pred = clf.predict(features_test)


# ANALYSE
export_graphviz(clf, out_file='tree.dot',
                class_names=["yes", "no"],
                rounded=True, proportion=False,
                precision=2, filled=True)

call(['dot', '-Tpng', 'tree.dot', '-o', './tree_5_b.png', '-Gdpi=600'])
print("Accuracy: ", metrics.accuracy_score(labels_test, labels_pred))
skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True)

plt.show()








