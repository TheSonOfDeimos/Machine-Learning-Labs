from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import scikitplot as skplt
from file_processor import *

# SETUP
features, labels = getFeaturesLabels("data/nn_1.csv")
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.20, random_state=42, shuffle=True)

# SCALING
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

solver_arr = ["lbfgs", "sgd", "adam"]

for solver in solver_arr:
        fig, sub = plt.subplots(2, 2)
        ax = sub.flatten()
        counter = 0
        for epoch in (100, 100, 100, 100) :
                # TRAINING
                mlp = MLPClassifier(hidden_layer_sizes=(15), max_iter=100, verbose=10, activation="relu", solver="lbfgs", alpha=0.01)
                #mlp = MLPClassifier(hidden_layer_sizes=(1), max_iter=epoch, activation=activation
                mlp.fit(features_train, labels_train.values.ravel())

                # TESTING
                labels_pred = mlp.predict(features_test)
                title = "Activation = relu" + "Solver = lbfgs" + " Epochs = " + str(epoch)
                disp = skplt.metrics.plot_confusion_matrix(labels_test, labels_pred, normalize=True, title=title, ax=ax[counter])
                counter += 1
                print(str(epoch))
        plt.show()
