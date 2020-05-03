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

#===================== Default settings ====================
# TRAIN
clf = tree.DecisionTreeClassifier()
classifier = clf.fit(features_train, labels_train)

# TEST



#===================== Non default settings ====================
best_set = ['1', 0, 0, 0.0]
best_clf = tree.DecisionTreeClassifier()
for split in ('best', 'random'):
    for depth in range(1, 20):
        for features in range(1, 7):
            new_clf = tree.DecisionTreeClassifier(splitter=split, max_depth=depth, max_features=features)
            new_clf.fit(features_train, labels_train)
            acc = new_clf.score(features_test, labels_test)
            if (best_set[3] < acc):
                best_set = [split, depth, features, acc]
                best_clf = new_clf
            if (acc > 0.8):
                print("Split = ", split, " Depth = ", depth, "Features = ", features, " Accuracy: ", acc)





# ANALYSE
new_clf = best_clf
acc = new_clf.score(features_test, labels_test)
labels_pred = new_clf.predict(features_test)

export_graphviz(new_clf, out_file='tree.dot',
                class_names=["yes", "no"],
                rounded=True, proportion=False,
                precision=2, filled=True)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_5_a.png', '-Gdpi=600'])



print("\n[BEST]  Split = ", best_set[0], " Depth = ", best_set[1], "Features = ", best_set[2], " Accuracy: ", acc)
skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True)

plt.show()








