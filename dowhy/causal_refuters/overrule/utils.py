"""Utilities for learning boolean rules.

This module implements the boolean ruleset estimator from OverRule [1]. Code is adapted (with some simplifications)
from https://github.com/clinicalml/overlap-code, under the MIT License.

[1] Oberst, M., Johansson, F., Wei, D., Gao, T., Brat, G., Sontag, D., & Varshney, K. (2020). Characterization of
Overlap in Observational Studies. In S. Chiappa & R. Calandra (Eds.), Proceedings of the Twenty Third International
Conference on Artificial Intelligence and Statistics (Vol. 108, pp. 788–798). PMLR. https://arxiv.org/abs/1907.04138
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def sampleUnif(x, n: int = 10000, seed: Optional[int] = None):
    """
    Generate samples from a uniform distribution over the max / min of each column of the sample X.

    These are used for estimation of support, as the number of samples included under the rules gives a measure of
    volume.  This function is specialized to continuous variables, while `sample_reference` handles the general case,
    calling this function where necessary.

    :param x: 2D array of samples, where each column corresponds to a feature.
    :type x: Pandas Dataframe or Numpy Array
    :param n: int, defaults to 10000
    :type n: int, optional
    :param seed: Random seed for uniform sampling, defaults to None
    :type seed: int, optional
    """
    rng = np.random.default_rng(seed)

    xMin, xMax = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
    refSamples = rng.uniform(low=xMin.tolist(), high=xMax.tolist(), size=(n, xMin.shape[0]))

    assert refSamples.shape[1] == x.shape[1]
    return refSamples


def sample_reference(
    x, n: Optional[int] = None, cat_cols: List[str] = [], seed: Optional[int] = None, ref_range: Optional[Dict] = None
):
    """
    Generate samples from a uniform distribution over the columns of X.

    :param x: 2D array of samples, where each column corresponds to a feature.
    :type x: Pandas Dataframe or Numpy Array
    :param n: Number of samples to draw, defaults to the same number as the samples provided.
    :type n: Optional[int], optional
    :param cat_cols: Set of categorical columns, defaults to None
    :type cat_cols: List[str], optional
    :param seed: Random seed for uniform sampling, defaults to None
    :type seed: int, optional
    :param ref_range: Manual override of the range for reference samples, given as a dictionary of the form
        `ref_range = {c: {"is_binary": True/False, "min": min_value, "max": max_value}}`
    :type ref_range: Optional[Dict], optional
    """

    if n is None:
        n = x.shape[0]

    rng = np.random.default_rng(seed)

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
                ref_cols[c] = rng.choice([0, 1], n)
            else:
                ref_cols[c] = rng.uniform(low=ref_range[c]["min"], high=ref_range[c]["max"], size=(n, 1)).ravel()
        else:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant column
            if valUniq < 2:
                ref_cols[c] = np.array([data[c].values[0]] * n)

            # Binary column
            elif valUniq == 2 or (c in cat_cols) or (data[c].dtype == "object"):
                cs = data[c].unique()
                ref_cols[c] = rng.choice(cs, n)

            # Ordinal column (seed = counter so not correlated)
            elif np.issubdtype(data[c].dtype, np.dtype(int).type) | np.issubdtype(data[c].dtype, np.dtype(float).type):
                ref_cols[c] = sampleUnif(data[[c]].values, n, seed=counter).ravel()
                if counter is not None:
                    counter += 1

    return pd.DataFrame(ref_cols)


def fatom(f: str, o: str, v: Optional[Union[str, float]], fmt: str = "%.3f") -> str:
    """
    Format an "atom", i.e., a single literal in a Boolean Rule.

    :param f: Feature name
    :type f: str
    :param o: Operator, one of ["<=", ">", ">=", "<", "==", "not", ""]
    :type o: str
    :param v: Value of comparison for ["<=", ">", ">=", "<", "=="]
    :type v: Optional[Union[str, float]]
    :param fmt: Formatting string for floats, defaults to "%.3f"
    :type fmt: str
    :return: Formatted atom
    :rtype: str
    """
    if o in ["<=", ">", ">=", "<", "=="]:
        if isinstance(v, str):
            return ("[%s %s %s]") % (f, o, v)
        else:
            return ("[%s %s " + fmt + "]") % (f, o, v)
    elif o == "not":
        return "not %s" % f
    else:
        return f


def rule_str(C: List, fmt: str = "%.3f") -> str:
    """
    Convert a rule into a string.

    :param C: List of rules, where each element is a list (a single rule) containing a set of atoms.
    :type C: List
    :param fmt: Formatting string for floats, defaults to "%.3f"
    :type fmt: str
    :return: Formatted rule
    :rtype: str
    """
    s = "  " + "\n∨ ".join(["(%s)" % (" ∧ ".join([fatom(a[0], a[1], a[2], fmt=fmt) for a in c])) for c in C])
    return s
