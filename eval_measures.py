import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

print "EMS reloaded"
def precision(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.

    Analagous to (absence of) Type I errors. Probability that a randomly
    selected document is classified correctly. I.e., no false negatives.
    """
    tn, fp, fn, tp = map(float, pred_table.flatten())# WRT to 1 as positive label
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return np.nan


def recall(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.

    Analagous to (absence of) Type II errors. Out of all the ones that are
    true, how many did you predict as true. I.e., no false positives.
    """
    tn, fp, fn, tp = map(float, pred_table.flatten())# WRT to 1 as positive label
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return np.nan


def accuracy(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.
    """
    tn, fp, fn, tp = map(float, pred_table.flatten())# WRT to 1 as positive label
    return (tp + tn) / (tp + tn + fp + fn)


def fscore_measure(pred_table, b=1):
    """
    For b, 1 = equal importance. 2 = recall is twice important. .5 recall is
    half as important, etc.
    """
    r = recall(pred_table)
    p = precision(pred_table)
    try:
        return (1 + b**2) * r*p/(b**2 * p + r)
    except ZeroDivisionError:
        return np.nan

def rmse(y_pred,y):
  return np.mean((y_pred-y)**2)**0.5

def mae(y_pred,y):
  return np.mean(np.abs(y_pred-y))

def auc(y_pred,y):
  return roc_auc_score(y, y_pred)


def roc(y_pred,y, float_precision=3):
  fpr, tpr, thresholds = roc_curve(y,
      np.round(y_pred, float_precision), pos_label=1)
  return tpr, fpr

def prc(y_pred,y, float_precision=3):
  precision, recall, thresholds = precision_recall_curve(y, np.round(y_pred, float_precision))
  return precision, recall

def prf1(y_pred, y, threshold=0.5):
  precision, recall, f1_score, support = precision_recall_fscore_support(y, y_pred > threshold, pos_valaverage='binary')
  return precision, recall, f1_score, support

def cm(y_pred, y, threshold=0.5, labels=[0,1]):
  return confusion_matrix(y, y_pred > threshold, labels=labels)


def show_all(pred_table):
  print "%10s\t%10s\t%10s\t%10s" % ("Precision", "Recall", "Accuracy", "F1-Score")
  print "%10s\t%10s\t%10s\t%10s" % (precision(pred_table), recall(pred_table), accuracy(pred_table), fscore_measure(pred_table))
