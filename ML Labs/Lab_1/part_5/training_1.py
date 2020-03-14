from subprocess import call
from IPython.display import Image
import numpy
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

from common.file_processor_glass import *
from subprocess import call
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz


# SETUP
features, labels = return_features_labels()
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, train_size=0.8, random_state=42, shuffle=True)

#===================== Default settings ====================
# TRAIN
clf = tree.DecisionTreeClassifier()
classifier = clf.fit(features_train, labels_train)

# TEST
labels_pred = clf.predict(features_test)


#===================== Non default settings ====================
for split in ('best', 'random'):
    for depth in range(1, 20):
        new_clf = tree.DecisionTreeClassifier(splitter=split, max_depth=depth)
        new_clf.fit(features_train, labels_train)
        print("Split = ", split, " Depth = ", depth, " Accuracy: ", new_clf.score(features_test, labels_test))




# ANALYSE
export_graphviz(clf, out_file='tree.dot',
                class_names=["1", "2", "3", "5", "6", "7"],
                rounded=True, proportion=False,
                precision=2, filled=True)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_5_a.png', '-Gdpi=600'])

print("Accuracy: ", metrics.accuracy_score(labels_test, labels_pred))
skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True)

plt.show()








