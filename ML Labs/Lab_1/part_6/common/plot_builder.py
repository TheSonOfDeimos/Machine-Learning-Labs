
import matplotlib.pyplot as plt_svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class PlotBuilder:
    def __init__(self):
        self.x_val_ = []
        self.pescision_score_ = []
        self.recall_score_ = []
        self.f1_score_ = []

    def appendData(self, x_val, pescision_score, recall_score, f1_score):
        self.x_val_.append(x_val)
        self.pescision_score_.append(pescision_score)
        self.recall_score_.append(recall_score)
        self.f1_score_.append(f1_score)

    def plot_summary_metricts(self):
        plt_svm.figure()

        # Percision score
        plt_svm.subplot(223)
        plt_svm.plot(self.x_val_, self.pescision_score_, '-o')
        plt_svm.yscale('linear')
        plt_svm.title('Percision score')
        plt_svm.grid(True)

        # Recall score
        plt_svm.subplot(224)
        plt_svm.plot(self.x_val_, self.recall_score_, '-o')
        plt_svm.yscale('linear')
        plt_svm.title('Recall score')
        plt_svm.grid(True)

        # F1 score
        plt_svm.subplot(221)
        plt_svm.plot(self.x_val_, self.f1_score_, '-o')
        plt_svm.yscale('linear')
        plt_svm.title('F1 score')
        plt_svm.grid(True)

        plt_svm.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                                wspace=0.35)

    def make_meshgrid(self, x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_svm(self, ax, clf, fetures_test, labels_test):
        X0, X1 = fetures_test[:, 0], fetures_test[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(
            xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=labels_test,
                    cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            

        
