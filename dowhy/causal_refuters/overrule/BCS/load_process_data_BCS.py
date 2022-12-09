# -*- coding: utf-8 -*-
# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Dennis Wei                          #
# ----------------------------------------------#

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

###########################################
#%% Top-level function
def load_process_data(filePath, rowHeader, colNames, colSep=',', fracPresent=0.9, col_y=None, valEq_y=None, colCateg=[], numThresh=9, negations=False):
    ''' Load CSV file and process data for BCS rule-learner

    Inputs:
    filePath = full path to CSV file
    rowHeader = index of row containing column names or None
    colNames = column names (if none in file or to override)
    colSep = column separator
    fracPresent = fraction of non-missing values needed to use a column (default 0.9)
    col_y = name of target column
    valEq_y = value to test for equality to binarize non-binary target column
    colCateg = list of names of categorical columns
    numThresh = number of quantile thresholds used to binarize ordinal variables (default 9)
    negations = whether to append negations

    Outputs:
    A = binary feature DataFrame
    y = target column'''

    # Read CSV file
    data = pd.read_csv(filePath, sep=colSep, names=colNames, header=rowHeader, error_bad_lines=False)

    # Remove columns with too many missing values
    data.dropna(axis=1, thresh=fracPresent * len(data), inplace=True)
    # Remove rows with any missing values
    data.dropna(axis=0, how='any', inplace=True)

    # Extract and binarize target column
    y = extract_target(data, col_y, valEq_y)

    # Binarize features
    A = binarize_features(data, colCateg, numThresh, negations)

    return A, y

#%% Extract and binarize target variable
def extract_target(data, col_y=None, valEq_y=None, valGt_y=None, **kwargs):
    '''Extract and binarize target variable

    Inputs:
    data = original feature DataFrame
    col_y = name of target column
    valEq_y = values corresponding to y = 1 for binarizing target
    valGt_y = threshold for binarizing target, above which y = 1

    Output:
    y = target column'''

    ### dmm: if no col_y specified -- use the last
    if not col_y and (col_y != 0):
        col_y = data.columns[-1]
    # Separate target column
    y = data.pop(col_y)
    if valEq_y or valEq_y == 0:
        # Binarize if values for equality test provided
        if type(valEq_y) is not list:
            valEq_y = [valEq_y]
        y = y.isin(valEq_y).astype(int)
    elif valGt_y or valGt_y == 0:
        # Binarize if threshold for comparison provided
        y = (y > valGt_y).astype(int)
    # Ensure y is binary and contains no missing values
    assert y.nunique() == 2, "Target 'y' must be binary"
    assert y.count() == len(y), "Target 'y' must not contain missing values"
    # Rename values to 0, 1
    y.replace(np.sort(y.unique()), [0, 1], inplace=True)

    return y

#%% Binarize features
def binarize_features(data, colCateg=[], numThresh=9, negations=False, threshStr=False, **kwargs):
    '''Binarize categorical and ordinal (including continuous) features

    Inputs:
    data = original feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)
    numThresh = number of quantile thresholds used to binarize ordinal variables (default 9)
    negations = whether to append negations
    threshStr = whether to convert thresholds on ordinal features to strings

    Outputs:
    A = binary feature DataFrame'''

    # Quantile probabilities
    quantProb = np.linspace(1./(numThresh + 1.), numThresh/(numThresh + 1.), numThresh)
    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    # Initialize dataframe and thresholds
    A = pd.DataFrame(index=data.index,
                     columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
    thresh = {}

    # Iterate over columns
    for c in data:
        # number of unique values
        valUniq = data[c].nunique()

        # Constant column --- discard
        if valUniq < 2:
            continue

        # Binary column
        elif valUniq == 2:
            # Rename values to 0, 1
            A[(str(c), '', '')] = data[c].replace(np.sort(data[c].unique()), [0, 1])
            if negations:
                A[(str(c), 'not', '')] = data[c].replace(np.sort(data[c].unique()), [1, 0])

        # Categorical column
        elif (c in colCateg) or (data[c].dtype == 'object'):
            # Dummy-code values
            if data[c].dtype == float:
                Anew = pd.get_dummies(data[c].astype(str)).astype(int)
            else:
                Anew = pd.get_dummies(data[c]).astype(int)
            Anew.columns = Anew.columns.astype(str)
            if negations:
                # Append negations
                Anew = pd.concat([Anew, 1-Anew], axis=1, keys=[(str(c),'=='), (str(c),'!=')])
            else:
                Anew.columns = pd.MultiIndex.from_product([[str(c)], ['=='], Anew.columns])
            # Concatenate
            A = pd.concat([A, Anew], axis=1)

        # Ordinal column
        elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
            | np.issubdtype(data[c].dtype, np.dtype(float).type):
            # Few unique values
            if valUniq <= numThresh + 1:
                # Thresholds are sorted unique values excluding maximum
                thresh[c] = np.sort(data[c].unique())[:-1]
            # Many unique values
            else:
                # Thresholds are quantiles excluding repetitions
                thresh[c] = data[c].quantile(q=quantProb).unique()
            # Threshold values to produce binary arrays
            Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
            if negations:
                # Append negations
                Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                ops = ['<=', '>']
            else:
                ops = ['<=']
            # Convert to dataframe with column labels
            if threshStr:
                Anew = pd.DataFrame(Anew, index=data.index,
                                    columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c].astype(str)]))
            else:
                Anew = pd.DataFrame(Anew, index=data.index,
                                    columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c]]))
            indNull = data[c].isnull()
            if indNull.any():
                # Ensure that rows corresponding to NaN values are zeroed out
                Anew.loc[indNull] = 0
                # Add NaN indicator column
                Anew[(str(c), '==', 'NaN')] = indNull.astype(int)
                if negations:
                    Anew[(str(c), '!=', 'NaN')] = (~indNull).astype(int)
            # Concatenate
            A = pd.concat([A, Anew], axis=1)

        else:
            print(("Skipping column '" + str(c) + "': data type cannot be handled"))
            continue

    return A

def binarize_categ(data, colCateg=[], **kwargs):
    '''Binarize categorical features only

    Inputs:
    data = original feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)

    Outputs:
    A = numeric feature DataFrame'''

    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    # Initialize dataframe and thresholds
    A = pd.DataFrame(index=data.index)

    # Iterate over columns
    for c in data:
        # number of unique values
        valUniq = data[c].nunique()

        # Constant column --- discard
        if valUniq < 2:
            continue

        # Binary column
        elif valUniq == 2:
            # Rename values to 0, 1
            A[str(c)] = data[c].replace(np.sort(data[c].unique()), [0, 1])

        # Categorical column
        elif (c in colCateg) or (data[c].dtype == 'object'):
            # Dummy-code values
            if data[c].dtype == float:
                Anew = pd.get_dummies(data[c].astype(str)).astype(int)
            else:
                Anew = pd.get_dummies(data[c]).astype(int)
            Anew.columns = str(c) + '==' + Anew.columns.astype(str)
            # Concatenate
            A = pd.concat([A, Anew], axis=1)

        # Ordinal column
        elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
            | np.issubdtype(data[c].dtype, np.dtype(float).type):
            # Leave as is
            A[str(c)] = data[c]

        else:
            print(("Skipping column '" + str(c) + "': data type cannot be handled"))
            continue

    return A

class FeatureBinarizer(TransformerMixin):
    '''Transformer for binarizing categorical and ordinal (including continuous) features
        Parameters:
            colCateg = list of categorical features ('object' dtype automatically treated as categorical)
            numThresh = number of quantile thresholds used to binarize ordinal variables (default 9)
            negations = whether to append negations
            threshStr = whether to convert thresholds on ordinal features to strings
            threshOverride = dictionary of {colname : np.linspace object} to define cuts
    '''
    def __init__(self, colCateg=[], numThresh=9, negations=False,
            threshStr=False, threshOverride={}, **kwargs):
        # List of categorical columns
        if type(colCateg) is pd.Series:
            self.colCateg = colCateg.tolist()
        elif type(colCateg) is not list:
            self.colCateg = [colCateg]
        else:
            self.colCateg = colCateg

        self.threshOverride = {} if threshOverride is None else threshOverride
        # Number of quantile thresholds used to binarize ordinal features
        self.numThresh = numThresh
        self.thresh = {}
        # whether to append negations
        self.negations = negations
        # whether to convert thresholds on ordinal features to strings
        self.threshStr = threshStr

    def fit(self, X):
        '''Inputs:
            X = original feature DataFrame
        Outputs:
            maps = dictionary of mappings for unary/binary columns
            enc = dictionary of OneHotEncoders for categorical columns
            thresh = dictionary of lists of thresholds for ordinal columns
            NaN = list of ordinal columns containing NaN values'''
        data = X
        # Quantile probabilities
        quantProb = np.linspace(1. / (self.numThresh + 1.), self.numThresh / (self.numThresh + 1.), self.numThresh)
        # Initialize
        maps = {}
        enc = {}
        thresh = {}
        NaN = []

        # Iterate over columns
        for c in data:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant or binary column
            if valUniq <= 2:
                # Mapping to 0, 1
                maps[c] = pd.Series(range(valUniq), index=np.sort(data[c].unique()))

            # Categorical column
            elif (c in self.colCateg) or (data[c].dtype == 'object'):
                # OneHotEncoder object
                enc[c] = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
                # Fit to observed categories
                enc[c].fit(data[[c]])

            # Ordinal column
            elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
                | np.issubdtype(data[c].dtype, np.dtype(float).type):
                # Few unique values
                if valUniq <= self.numThresh + 1:
                    # Thresholds are sorted unique values excluding maximum
                    thresh[c] = np.sort(data[c].unique())[:-1]
                # Many unique values
                elif c in self.threshOverride.keys():
                    thresh[c] = self.threshOverride[c]
                else:
                    # Thresholds are quantiles excluding repetitions
                    thresh[c] = data[c].quantile(q=quantProb).unique()
                if data[c].isnull().any():
                    # Contains NaN values
                    NaN.append(c)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        self.maps = maps
        self.enc = enc
        self.thresh = thresh
        self.NaN = NaN
        return self

    def transform(self, X):
        '''Inputs:
            X = original feature DataFrame
        Outputs:
            A = binary feature DataFrame'''
        data = X
        maps = self.maps
        enc = self.enc
        thresh = self.thresh
        NaN = self.NaN

        # Initialize dataframe
        A = pd.DataFrame(index=data.index,
                         columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))

        # Iterate over columns
        for c in data:
            # Constant or binary column
            if c in maps:
                # Rename values to 0, 1
                A[(str(c), '', '')] = data[c].map(maps[c])
                if self.negations:
                    A[(str(c), 'not', '')] = 1 - A[(str(c), '', '')]

            # Categorical column
            elif c in enc:
                # Apply OneHotEncoder
                Anew = enc[c].transform(data[[c]])
                Anew = pd.DataFrame(Anew, index=data.index, columns=enc[c].categories_[0].astype(str))
                if self.negations:
                    # Append negations
                    Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(str(c), '=='), (str(c), '!=')])
                else:
                    Anew.columns = pd.MultiIndex.from_product([[str(c)], ['=='], Anew.columns])
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            # Ordinal column
            elif c in thresh:
                # Threshold values to produce binary arrays
                Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
                if self.negations:
                    # Append negations
                    Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                    ops = ['<=', '>']
                else:
                    ops = ['<=']
                # Convert to dataframe with column labels
                if self.threshStr:
                    Anew = pd.DataFrame(Anew, index=data.index,
                                        columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c].astype(str)]))
                else:
                    Anew = pd.DataFrame(Anew, index=data.index,
                                        columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c]]))
                if c in NaN:
                    # Ensure that rows corresponding to NaN values are zeroed out
                    indNull = data[c].isnull()
                    Anew.loc[indNull] = 0
                    # Add NaN indicator column
                    Anew[(str(c), '==', 'NaN')] = indNull.astype(int)
                    if self.negations:
                        Anew[(str(c), '!=', 'NaN')] = (~indNull).astype(int)
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        return A

#%% Discretize continuous features and standardize values
def bin_cont_features(data, colCateg=[], numThresh=9, **kwargs):
    '''Bin continuous features using quantiles

    Inputs:
    data = original feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)
    numThresh = number of quantile thresholds not including min/max (default 9)

    Outputs:
    A = discretized feature DataFrame'''

    # Quantile probabilities
    quantProb = np.linspace(0., 1., numThresh+2)
    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    # Initialize DataFrame
    A = data.copy()
    # Iterate over columns
    for c in data:
        # number of unique values
        valUniq = data[c].nunique()

        # Only bin non-categorical numerical features with enough unique values
        if (np.issubdtype(data[c].dtype, np.dtype(int).type) \
        or np.issubdtype(data[c].dtype, np.dtype(float).type))\
        and (c not in colCateg) and valUniq > numThresh + 1:
            A[c] = pd.qcut(A[c], q=quantProb, duplicates='drop')
            if A[c].nunique() == 1:
                # Collapsed into single bin, re-separate into two bins
                quant = data[c].quantile([0, 0.5, 1])
                quant[0] -= 1e-3
                if quant[0.5] == quant[1]:
                    quant[0.5] -= 1e-3
                A[c] = pd.cut(data[c], quant)

    return A

def std_values(data, colCateg=[], **kwargs):
    '''Standardize values of (already discretized) features

    Inputs:
    data = input feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)

    Outputs:
    A = standardized feature DataFrame
    mappings = dictionary of value mappings'''

    # Initialize
    A = data.copy()
    mappings = {}
    isCategory = A.dtypes == 'category'
    # Iterate over columns
    for c in A:
        # number of unique values
        valUniq = A[c].nunique()

        # Binned numerical column, which has 'category' dtype
        if isCategory[c]:
            # Map bins to integers
            mappings[c] = pd.Series(range(valUniq), index=A[c].cat.categories)
            A[c].cat.categories = range(valUniq)

        # Binary column
        elif valUniq == 2:
            # Map sorted values to 0, 1
            mappings[c] = pd.Series([0, 1], index=np.sort(A[c].dropna().unique()))
            A[c] = A[c].map(mappings[c])

        # Categorical column
        elif (c in colCateg) or (A[c].dtype == 'object'):
            # First map sorted values to integers
            mappings[c] = pd.Series(range(valUniq), index=np.sort(A[c].dropna().unique()))
            # Then map to alphabetic encoding of integers
            mappings[c] = mappings[c].map(digit_to_alpha)
            A[c] = A[c].map(mappings[c])

        # Non-binned numerical column (because it has few unique values)
        elif np.issubdtype(A[c].dtype, np.dtype(int).type) \
            or np.issubdtype(A[c].dtype, np.dtype(float).type):
            # Map sorted values to integers
            mappings[c] = pd.Series(range(valUniq), index=np.sort(A[c].dropna().unique()))
            A[c] = A[c].map(mappings[c])

    return A, mappings

def digit_to_alpha(n):
    '''Map digits in integer n to letters
    0 -> A, 1 -> B, 2 -> C, ..., 9 -> J'''
    return ''.join([chr(int(d) + ord('A')) for d in str(n)])

#%% Split data into training and test sets
def split_train_test(A, y, dirData, fileName, numFold=10, numRepeat=10, concatMultiIndex=False):
    '''Split data into training and test sets using repeated stratified K-fold CV
    and save as CSV

    Inputs:
    A = binary feature DataFrame
    y = target column
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    numFold = number of folds (K)
    numRepeat = number of K-fold splits
    concatMultiIndex = whether to concatenate column MultiIndex into single level

    Output: total number of splits = numFold x numRepeat'''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(y.name,'','')] = y
    else:
        B[y.name] = y
    # Concatenate column MultiIndex into single level
    if concatMultiIndex and (type(B.columns) is pd.MultiIndex):
        B.columns = B.columns.get_level_values(0) + B.columns.get_level_values(1) + B.columns.get_level_values(2)
    # Iterate over splits
    rskf = RepeatedStratifiedKFold(n_splits=numFold, n_repeats=numRepeat)
    for (split, (idxTrain, idxTest)) in enumerate(rskf.split(A, y)):
        # Save training and test sets as CSV
        filePath = os.path.join(dirData, fileName + '_' + format(split, '03d') + '_')
        B.iloc[idxTrain].to_csv(filePath + 'train.csv', index=False)
        B.iloc[idxTest].to_csv(filePath + 'test.csv', index=False)

    return rskf.get_n_splits()

def save_train_test(A, y, splits, dirData, fileName, concatMultiIndex=False):
    '''Save training and test sets as CSV if given splits, otherwise full dataset

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    concatMultiIndex = whether to concatenate column MultiIndex into single level
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y
    # Concatenate column MultiIndex into single level
    if concatMultiIndex and (type(B.columns) is pd.MultiIndex):
        if concatMultiIndex == 'BOA':
            # Special formatting for Wang et al.'s Bayesian Rule Sets
            B.columns = B.columns.get_level_values(0).str.replace('_','-') + '_'\
            + B.columns.get_level_values(1) + B.columns.get_level_values(2)
        else:
            B.columns = B.columns.get_level_values(0) + B.columns.get_level_values(1) + B.columns.get_level_values(2)

    if splits:
        for (split, (idxTrain, idxTest)) in enumerate(splits):
            # Save training and test sets as CSV
            filePath = os.path.join(dirData, fileName + '_' + format(split, '03d') + '_')
            B.iloc[idxTrain].to_csv(filePath + 'train.csv', index=False)
            B.iloc[idxTest].to_csv(filePath + 'test.csv', index=False)
    else:
        # Save full dataset
        filePath = os.path.join(dirData, fileName + '.csv')
        B.to_csv(filePath, index=False)

    return

def pickle_train_test(A, y, splits, dirData, fileName):
    '''Pickle training and test sets if given splits, otherwise full dataset

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y

    if splits:
        for (split, (idxTrain, idxTest)) in enumerate(splits):
            # Pickle training and test sets
            filePath = os.path.join(dirData, fileName + '_' + format(split, '03d') + '_')
            B.iloc[idxTrain].to_pickle(filePath + 'train.pkl')
            B.iloc[idxTest].to_pickle(filePath + 'test.pkl')
    else:
        # Pickle full dataset
        filePath = os.path.join(dirData, fileName + '.pkl')
        B.to_pickle(filePath)

    return

def save_internal_train(A, y, splits, dirData, fileName, concatMultiIndex=False):
    '''Save internal training sets as CSV for parameter selection

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    concatMultiIndex = whether to concatenate column MultiIndex into single level
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y
    # Concatenate column MultiIndex into single level
    if concatMultiIndex and (type(B.columns) is pd.MultiIndex):
        if concatMultiIndex == 'BOA':
            # Special formatting for Wang et al.'s Bayesian Rule Sets
            B.columns = B.columns.get_level_values(0).str.replace('_','-') + '_'\
            + B.columns.get_level_values(1) + B.columns.get_level_values(2)
        else:
            B.columns = B.columns.get_level_values(0) + B.columns.get_level_values(1) + B.columns.get_level_values(2)

    # Iterate over test set index
    numSplit = len(splits)
    with open(os.path.join(dirData, 'splitsInt.txt'), 'w') as f:
        for (test, (idxTrain, idxTest)) in enumerate(splits):
            # Iterate over validation set index
            for valid in range(test+1, numSplit):
                # Internal training set indices
                idxIntTrain = np.setdiff1d(idxTrain, splits[valid][1])
                # Save as CSV
                filePath = os.path.join(dirData, fileName + '_' + format(test, '03d') + '_'\
                                        + format(valid, '03d') + '_train.csv')
                B.iloc[idxIntTrain].to_csv(filePath, index=False)
                # Write training, test, and validation indices to text file
                f.write(str(idxIntTrain).strip('[]').replace('\n','') + '\n')
                f.write(str(idxTest).strip('[]').replace('\n','') + '\n')
                f.write(str(splits[valid][1]).strip('[]').replace('\n','') + '\n')

    return

def pickle_internal_train(A, y, splits, dirData, fileName):
    '''Pickle internal training sets for parameter selection

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y

    # Iterate over test set index
    numSplit = len(splits)
    for (test, (idxTrain, idxTest)) in enumerate(splits):
        # Iterate over validation set index
        for valid in range(test+1, numSplit):
            # Internal training set indices
            idxIntTrain = np.setdiff1d(idxTrain, splits[valid][1])
            # Save as CSV
            filePath = os.path.join(dirData, fileName + '_' + format(test, '03d') + '_'\
                                    + format(valid, '03d') + '_train.pkl')
            B.iloc[idxIntTrain].to_pickle(filePath)

    return


#%% Call function if run as script
##########################################
def example():

    # File parameters
    workDir = u''
    fileDir = './Data/' #u'\\Data\\'
    fileName = u'iris_categ.csv'
    colSep = ','
    rowHeader = None
    colNames = ['X1','X2','sepal length','sepal width','petal length','petal width','X7','X8','iris species']
    col_y = colNames[-1]
    colCateg = 'X8' # ['X8']

    A, y = load_process_data(workDir + fileDir + fileName, rowHeader, colNames, colSep=colSep, col_y=col_y, valEq_y=2, colCateg=colCateg)
###########################################
def example2():

    fname_data = 'Data/iris_bin.csv'
    colSep = ','
    rowHeader = None
    colNames = ['X1', 'X2', 'sepal length', 'sepal width', 'petal length', 'petal width', 'iris species']
    col_y = colNames[-1]
    A, y = load_process_data(fname_data, rowHeader, colNames, colSep=colSep, col_y=col_y, valEq_y=2)

###########################################
if __name__ == '__main__':
    #example2()
    example()
