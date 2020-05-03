import matplotlib.pyplot as plt
import numpy as np


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

    def show(self, figure_name):
        fig, sub = plt.subplots(1, 3)
        ax = sub.flatten()

        ax[0].plot(self.x_val_, self.pescision_score_, 'o-')
        ax[0].set_title('Precision score')
        ax[0].grid(True)
        ax[0].set_xlabel("Estimators number")

        # Recall score
        ax[1].plot(self.x_val_, self.recall_score_, 'o-')
        ax[1].set_title('Recall score')
        ax[1].grid(True)
        ax[1].set_xlabel("Estimators number")

        # F1 score
        ax[2].plot(self.x_val_, self.f1_score_, 'o-')
        ax[2].set_title('F1 score')
        ax[2].grid(True)
        ax[2].set_xlabel("Estimators number")

        fig.suptitle(figure_name)

        plt.show()
