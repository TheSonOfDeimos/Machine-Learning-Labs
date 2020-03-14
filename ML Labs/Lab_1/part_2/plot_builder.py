import matplotlib.pyplot as plt_svm
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

        def show(self):
                plt_svm.figure()

                # Percision score
                plt_svm.subplot(223)
                plt_svm.plot(self.x_val_, self.pescision_score_, 'o-')
                plt_svm.yscale('linear')
                plt_svm.title('Percision score')
                plt_svm.grid(True)
                plt_svm.xticks(np.arange(min(self.x_val_),
                                     max(self.x_val_)+1, 1.0))

                # Recall score
                plt_svm.subplot(224)
                plt_svm.plot(self.x_val_, self.recall_score_, 'o-')
                plt_svm.yscale('linear')
                plt_svm.title('Recall score')
                plt_svm.grid(True)
                plt_svm.xticks(np.arange(min(self.x_val_),
                                     max(self.x_val_)+1, 1.0))

                # F1 score
                plt_svm.subplot(221)
                plt_svm.plot(self.x_val_, self.f1_score_, 'o-')
                plt_svm.yscale('linear')
                plt_svm.title('F1 score')
                plt_svm.grid(True)
                plt_svm.xticks(np.arange(min(self.x_val_),
                                     max(self.x_val_)+1, 1.0))

                plt_svm.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                                    wspace=0.35)

                plt_svm.show()
                                

