import numpy as np


def accuracy(y_test, y_pred):
    return np.sum(np.equal(y_test, y_pred)) / len(y_test)


def precision(y_test, y_pred):
    mask = y_pred == 1
    true_positive, false_positive = __tp_fp_tn_fn(y_test, y_pred, mask)
    precision_sc = true_positive / (true_positive + false_positive)
    return precision_sc


def recall(y_test, y_pred):
    mask = y_test == 1
    true_positive, false_negative = __tp_fp_tn_fn(y_test, y_pred, mask)
    recall_sc = true_positive / (true_positive + false_negative)
    return recall_sc


def f1_score(y_test, y_pred):
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    f1 = 2 * (p * r / (p + r))
    f1_score(y_test, y_pred)
    return f1


def __tp_fp_tn_fn(y_test, y_pred, mask):
    test_posneg = y_test[mask]
    pred_posneg = y_pred[mask]
    tp_tn = np.sum(np.equal(test_posneg, pred_posneg))
    fp_fn = len(pred_posneg) - tp_tn
    return tp_tn, fp_fn
