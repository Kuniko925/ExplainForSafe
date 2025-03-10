import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.spatial import distance

class CoverageMetrics:
    def __init__(self, ground_truth, prediction):
        self.ground_truth = ground_truth.astype(int).ravel()
        self.prediction = prediction.ravel()
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.ground_truth, self.prediction).ravel()
        
    def intersection_over_union(self):
        return self.tp / (self.tp + self.fp + self.fn)

    def dice_coefficient(self):
        return 2*self.tp / (2*self.tp + self.fp + self.fn)

    def sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def specificity(self):
        return self.tn / (self.tn + self.fp)

    def pixel_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def area_under_curve(self):
        return roc_auc_score(self.ground_truth, self.prediction)

    def cohens_kappa(self):
        fc = ((self.tn+self.fn)*(self.tn+self.fp) + (self.fp+self.tp)*(self.fn+self.tp)) / (self.tp + self.tn + self.fp + self.fn)
        return ((self.tp+self.tn) - fc) / ((self.tp + self.tn + self.fp + self.fn) - fc)

    def false_negative_rate(self):
        return self.fn / (self.fn + self.tp)
    
    def false_positive_rate(self):
        return self.fp / (self.fp + self.tn)
