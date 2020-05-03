from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import scikitplot as skplt
from fileProcesor_titanic import *
from plot_builder import *



# Prepare dataframes
features, labels = return_features_labels("data/titanic.csv")
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.8, random_state=42, shuffle=True)

# Init estimators
# Create Learners per layer
layer_one_estimators = [
    ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('knn_1', KNeighborsClassifier(n_neighbors=5))]
    
layer_two_estimators = [
        ('dt_2', DecisionTreeClassifier()),
        ('rf_2', RandomForestClassifier(n_estimators=50, random_state=42)),]

layer_two = StackingClassifier(estimators=layer_two_estimators, final_estimator=LogisticRegression())

# Create Final model by
model = StackingClassifier(estimators=layer_one_estimators, final_estimator=layer_two)

model.fit(features_train, labels_train)
labels_pred = model.predict(features_test)

skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True)
plt.show()
