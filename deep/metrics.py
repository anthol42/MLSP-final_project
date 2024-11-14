import torch
from sklearn.metrics import precision_score, accuracy_score
import numpy as np

def accuracy(targets, pred):
    hard_pred = np.argmax(pred, axis=1)
    return accuracy_score(targets, hard_pred)

def custom_precision(targets, pred):
    hard_pred = np.argmax(pred, axis=1)
    # Down
    btargets = targets == 0
    bhard_pred = hard_pred == 0
    down_prec = precision_score(btargets, bhard_pred, zero_division=0)

    # UP
    btargets = targets == 1
    bhard_pred = hard_pred == 1
    up_prec = precision_score(btargets, bhard_pred, zero_division=0)

    prec = (up_prec + down_prec) / 2
    return prec

def precision_2d(targets, pred):
    """
    :param targets: Shape (B, ) # 0: Down, 1: Up
    :param pred: Shape (B, 2)
    :return: Scalar tensor
    """
    hard_pred = np.argmax(pred, axis=1)
    return precision_score(targets, hard_pred, zero_division=0)
