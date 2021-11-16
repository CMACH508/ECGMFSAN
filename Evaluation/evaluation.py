from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import torch
import matplotlib as mpl
from time import ctime
from threading import Thread
from multiprocessing import Pool
FloatFormat='%.2f'

# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)
    #print(A)
    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    print(f_measure)
    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure

# Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
def compute_beta_measures(labels, outputs, beta):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1+beta**2)*tp + fp + beta**2*fn:
            f_beta_measure[k] = float((1+beta**2)*tp) / float((1+beta**2)*tp + fp + beta**2*fn)
        else:
            f_beta_measure[k] = float('nan')
        if tp + fp + beta*fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta*fn)
        else:
            g_beta_measure[k] = float('nan')

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):

        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)

        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return macro_auroc, macro_auprc

# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score
# For each set of equivalent classes, replace each class with the representative class for the set.
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                classes[j] = multiple_classes[0] # Use the first class as the representative class.
    return classes

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False
# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values
def load_weights(weight_file, equivalent_classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)

    # For each collection of equivalent classes, replace each class with the representative class for the set.
    rows = replace_equivalent_classes(rows, equivalent_classes)

    # Check that equivalent classes have identical weights.
    for j, x in enumerate(rows):
        for k, y in enumerate(rows[j+1:]):
            if x==y:
                assert(np.all(values[j, :]==values[j+1+k, :]))
                assert(np.all(values[:, j]==values[:, j+1+k]))

    # Use representative classes.
    classes = [x for j, x in enumerate(rows) if x not in rows[:j]]
    indices = [rows.index(x) for x in classes]
    weights = values[np.ix_(indices, indices)]

    return classes, weights
def Evaluation(labels,scalar_outputs,binary_outputs):

    thresholds=ThresholdsGridSearch(labels,scalar_outputs)
    #print(thresholds)
    binary_outputs1=scalar_outputs>thresholds
    auroc, auprc = compute_auc(labels, scalar_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)
    f_beta_measure, g_beta_measure = compute_beta_measures(labels, binary_outputs, beta=2)


    # # calculate challenge score
    # weights_file = 'weights.csv'
    # normal_class = '426783006'
    # equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    # classes, weights = load_weights(weights_file, equivalent_classes)  # 从官方weight文件里读取
    # challenge_metric0 = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
    # challenge_metric1 = compute_challenge_metric(weights, labels, binary_outputs1, classes, normal_class)

    print("auroc:{:.2f} auprc:{:.2f} f_measure:{:.2f} f_beta_measure:{:.2f} g_beta_measure:{:.2f} ".format(auroc, auprc,
                                                                                                           f_measure,
                                                                                                           f_beta_measure,
                                                                                                           g_beta_measure))
    return auroc,auprc,f_measure,f_beta_measure,g_beta_measure#,challenge_metric

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
def ThresholdSearch(labels,prob_outputs,thresholds,i,weights,classes,normal_class,thresholds_num=101):
    max_score=0
    for threshold in np.linspace(0, 1, thresholds_num):
        threshold_before = thresholds[i]
        thresholds[i] = threshold
        binary_outputs = (prob_outputs >= thresholds).astype('int')
        challenge_score = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
        if challenge_score < max_score:
            thresholds[i] = threshold_before
        else:
            max_score = challenge_score
    return thresholds[i]

def get_challenge_score(labels,prob_outputs,thresholds=None):
    weights_file = '/home/dengfucheng/ECG_project/challenge2020/Evaluation/weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    classes, weights = load_weights(weights_file, equivalent_classes)  # 从官方weight文件里读取

    thresholds=thresholds if thresholds else 0.5
    binary_outputs = (prob_outputs >= thresholds).astype('int')
    challenge_score = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)

    return challenge_score


def ThresholdsGridSearch(labels,prob_outputs):
    '''
    search the thresholds
    first,search the optimal value for all classes
    then , search thresholds for every class
    '''
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    classes, weights = load_weights(weights_file, equivalent_classes)  # 从官方weight文件里读取
    # common stage
    max_score=0
    max_common_threshold=0.0
    common_thresholds=np.linspace(0,1,11)
    for common_threshold in common_thresholds:
        binary_outputs=(prob_outputs>=common_threshold).astype('int')
        challenge_score = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
        if challenge_score>max_score:
            max_score=challenge_score
            max_common_threshold=common_threshold
    binary_outputs = (prob_outputs >= max_common_threshold).astype('int')
    challenge_score = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
    print('stage 0 challenge score:', challenge_score)

    #first stage ,search for optimal thresholds with step 0.1
    thresholds=np.ones(len(classes))*max_common_threshold
    p=Pool(24)
    res_l=[]
    for i in range(len(classes)):
        res=p.apply_async(ThresholdSearch,args=(labels,prob_outputs,thresholds,i,weights,classes,normal_class,11,))
        res_l.append(res)
    #for i in range(len(classes)):
    thresholds = np.array([x.get() for x in res_l] )
    print('first stage thresholds:',thresholds)
    binary_outputs = (prob_outputs >= thresholds).astype('int')
    challenge_score = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
    print('first stage challenge score:', challenge_score)

    # second stage, search for optimal thresholds with step 0.01
    #thresholds = np.ones(len(classes)) * max_common_threshold
    p=Pool(24)
    res_l=[]
    for i in range(len(classes)):
        res=p.apply_async(ThresholdSearch,args=(labels,prob_outputs,thresholds,i,weights,classes,normal_class))
        res_l.append(res)
    #for i in range(len(classes)):
    thresholds = np.array([x.get() for x in res_l] )
    print('second stage thresholds:',thresholds)



    binary_outputs = (prob_outputs >= thresholds).astype('int')
    challenge_score = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
    Evaluation(labels,prob_outputs,binary_outputs)
    print('second stage challenge score:',challenge_score)
    #print('Current thresholds:',thresholds)
    return thresholds


def test():
    probs=np.array([[0.9,0.7],[0.6,0.9]])
    th=0.6#[0.7,0.8]
    print((probs>th).astype('int'))

if __name__ == '__main__':
    #ThresholdsGridSearch(np.array([[0,1],[1,0]]),np.array([[0.3,0.5],[0.7,0.7]]))
    #ThresholdsSearch([[0,1]],[[0.3,0.5]])
    test()