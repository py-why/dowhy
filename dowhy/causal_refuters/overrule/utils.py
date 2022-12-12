# ----------------------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets              #
# @Authors: Fredrik D. Johansson, Michael Oberst, Tian Gao  #
# ----------------------------------------------------------#

import numpy as np
import pandas as pd


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
    refSamples = np.random.uniform(low=xMin.tolist(), high=xMax.tolist(), size=(n, xMin.shape[0]))

    assert refSamples.shape[1] == x.shape[1]
    return refSamples


def sample_reference(x, n=None, cat_cols=[], seed=None, ref_range=None):
    """Generates samples from a uniform distribution over the columns of X
    TODO: Docstring
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
            if ref_range[c]["is_binary"]:
                ref_cols[c] = np.random.choice([0, 1], n)
            else:
                ref_cols[c] = np.random.uniform(low=ref_range[c]["min"], high=ref_range[c]["max"], size=(n, 1)).ravel()
        else:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant column
            if valUniq < 2:
                ref_cols[c] = [data[c].values[0]] * n

            # Binary column
            elif valUniq == 2 or (c in cat_cols) or (data[c].dtype == "object"):
                cs = data[c].unique()
                ref_cols[c] = np.random.choice(cs, n)

            # Ordinal column (seed = counter so not correlated)
            elif np.issubdtype(data[c].dtype, np.dtype(int).type) | np.issubdtype(data[c].dtype, np.dtype(float).type):
                ref_cols[c] = sampleUnif(data[[c]].values, n, seed=counter).ravel()
                if counter is not None:
                    counter += 1

    return pd.DataFrame(ref_cols)


def fatom(f, o, v, fmt="%.3f"):
    if o in ["<=", ">", ">=", "<", "=="]:
        if isinstance(v, str):
            return ("[%s %s %s]") % (f, o, v)
        else:
            return ("[%s %s " + fmt + "]") % (f, o, v)
    elif o == "not":
        return "not %s" % f
    else:
        return f


def rule_str(C, fmt="%.3f"):
    s = "  " + "\n∨ ".join(["(%s)" % (" ∧ ".join([fatom(a[0], a[1], a[2], fmt=fmt) for a in c])) for c in C])
    return s
