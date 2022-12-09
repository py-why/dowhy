# ----------------------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets              #
# @Authors: Fredrik D. Johansson, Michael Oberst, Tian Gao  #
# ----------------------------------------------------------#

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, brier_score_loss, classification_report, roc_curve
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model
import logging

def sampleUnif(x, n=10000, seed=None):
    """Generates samples from a uniform distribution over the max / min of each
    column of the sample X

    @args:
        x: Samples as a 2D numpy array
        n: Number of samples to return

    @returns:
        refSamples: Uniform samples as numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    xMin, xMax = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
    refSamples = np.random.uniform(low=xMin.tolist(),
                                   high=xMax.tolist(),
                                   size=(n, xMin.shape[0]))

    assert(refSamples.shape[1] == x.shape[1])
    return refSamples

def log(path, str='', output=True, start=False):
    """ Log a string to a given path
    """
    if path is not None:
        if start:
            open(path, 'w').close()
        else:
            with open(path, 'a') as f:
                f.write(str+'\n')
    if output:
        print(str)

def sample_reference(x, n=None, cat_cols=[], seed=None, ref_range=None):
    """Generates samples from a uniform distribution over the columns of X

    @args:
        x: Samples as a 2D numpy array or pandas dataframe @TODO: implement
        n: Number of samples to return

    @returns:
        reference_samples: Uniform samples as numpy array
    """

    if n is None:
        n = x.shape[0]

    if seed is not None:
        np.random.seed(seed)

    data = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
    
    if ref_range is not None:
        assert isinstance(ref_range, dict)
    else:
        ref_range = {}

    ref_cols = {}
    counter = seed
    # Iterate over columns
    for c in data:
        if c in ref_range.keys():
            # logging.info("Using provided reference range for {}".format(c))
            if ref_range[c]['is_binary']:
                ref_cols[c] = np.random.choice([0, 1], n)
            else:
                ref_cols[c] = np.random.uniform(
                               low=ref_range[c]['min'],
                               high=ref_range[c]['max'],
                               size=(n, 1)).ravel()
        else:            
            # number of unique values
            valUniq = data[c].nunique()

            # Constant column
            if valUniq < 2:
                ref_cols[c] = [data[c].values[0]]*n

            # Binary column
            elif valUniq == 2 or (c in cat_cols) or (data[c].dtype == 'object'):
                cs = data[c].unique()
                ref_cols[c] = np.random.choice(cs, n)

            # Ordinal column (seed = counter so not correlated)
            elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
                | np.issubdtype(data[c].dtype, np.dtype(float).type):
                ref_cols[c] = sampleUnif(data[[c]].values, n, seed=counter).ravel()
                if counter is not None:
                    counter += 1

    return pd.DataFrame(ref_cols)

def compute_metrics(y_true, y_predicted, y_prob = None):
    """compute metrics for the prredicted labels against ground truth

        @args:
            y_true: the ground truth label
            y_predicted: the predicted label
            y_predicted_prob: probability of the predicted label

        @returns:
            various metrics: F1-score, AUC of ROC, brier-score, also plots AUC
    """

    # plot AUC
    if y_prob:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()

        # brier = brier_score_loss((y_true, y_prob))

    # F1 score and brier score
    f1 = f1_score(y_true, y_predicted)

    # classification report
    plot_classification_report(classification_report(y_true, y_predicted))

    return f1



def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):

    """plot classification report

            @args:
                classificationReport: sklearn classification report
        """

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()



def causal_treatment_effect(x, y, T):
    """Estimating treatment effects

            @args:
                   x: covariates, shape N by k
                   y: effects, shape N by 1
                   T: treatment, shape N by 1
    """

    # compute ATE directly using sample means

    # get groups
    index_T1 = np.where(T.flatten()==1)
    index_T0 = np.where(T.flatten()==0)

    # Method 1: ATE
    ATE = np.mean(y[index_T1]) - np.mean(y[index_T0])


    # Method 2: fit propensity model
    T1_model = linear_model.LinearRegression()
    T0_model = linear_model.LinearRegression()
    # T0_model.fit(np.reshape(x[index_T0], (-1, 1)), y[index_T0])
    T0_model.fit(x[index_T0], y[index_T0])
    T1_model.fit(x[index_T1], y[index_T1])

    ATE2 = np.mean(T1_model.predict(x)) - np.mean(T0_model.predict(x))


    return ATE, ATE2

def fatom(f, o, v, fmt='%.3f'):
    if o in ['<=', '>', '>=', '<', '==']:
        if isinstance(v, str):
            return ('[%s %s %s]') % (f,o,v)
        else:
            return ('[%s %s '+fmt+']') % (f,o,v)
    elif o == 'not':
        return 'not %s' % f
    else:
        return f

def rule_str(C, fmt='%.3f'):
    s = '  '+'\n∨ '.join(['(%s)' % (' ∧ '.join([fatom(a[0], a[1], a[2], fmt=fmt) for a in c])) for c in C])
    return s
