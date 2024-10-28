from collections import defaultdict
from math import log

import numpy as np
from scipy.stats import norm, t


def compute_ci(r=None, nx=None, ny=None, confidence=0.95):
    """Compute Parametric confidence intervals around correlation coefficient.
    See : https://online.stat.psu.edu/stat505/lesson/6/6.3

    This is done by applying Fisher's r to z transform
    z = .5[ln((1+r)/(1-r))] = arctanh(r)

    The Standard error is 1/sqrt(N-3) where N is sample size

    The critical value for normal distribution for a corresponding confidence
    level is calculated from stats.norm.ppf((1 - alpha)/2) for two tailed test

    The lower and upper condidence intervals in z space are calculated with the formula
    z ± critical value*error

    The confidence interval is then converted back to r space

    :param stat : correlation coefficient
    :param nx : length of vector x
    :param ny :length of vector y
    :param confidence : Confidence level (0.95 = 95%)

    :returns : array containing confidence interval
    """

    assert r is not None and nx is not None
    assert isinstance(confidence, float)
    assert 0 < confidence < 1

    z = np.arctanh(r)  # Fisher Transform  from r to z
    se = 1 / np.sqrt(nx - 3)  # Standard error = 1/sqrt(N-3) where N is sample size
    crit = np.abs(norm.ppf((1 - confidence) / 2))  # Z-critical value
    ci_z = np.array([z - crit * se, z + crit * se])  # CI = point estimator ± critical value*error
    ci = np.tanh(ci_z)  # Back Transform to r-space

    return ci


def partial_corr(data=None, x=None, y=None, z=None, method="pearson"):
    """Calculate Partial correlation which is the degree of association between
    x and y after removing effect of z. This is done by calculating correlation
    coefficient between the residuals of two linear regressions :
    x\sim z, y\sim z
    See : 1 https://en.wikipedia.org/wiki/Partial_correlation
          2 https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-842X.2004.00360.x?casa_token=p_D3joHC8C0AAAAA:qigIZHVfcVi8vsz1j2t7uQYOorrYaF3Tm4lpQOUzqG_J9gJgtFerOyliKBnQPVG187nJxbA-wcbXU3QcOw
          3 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4681537/
          4 http://parker.ad.siu.edu/Olive/slch6.pdf
    :param data : pandas dataframe
    :param x : Column name in data
    :param y : Column name in data
    :param z : string or list
    :param method : string denoting the correlation type - "pearson" or "spearman"

    : returns: a python dictionary with keys as
        n: Sample size
        r: Partial correlation coefficient
        CI95: 95% parametric confidence intervals
        p-val: p-value
    """

    assert data.shape[0] > 2  # Check for atleast 3 samples
    assert x != z  # x and z should be distinct
    assert y != z  # y and z should be distinct
    assert x != y  # x and y should be distinct
    if isinstance(z, list):
        assert x not in z  # x and z should be distinct
        assert y not in z  # y and z should be distinct

    combined_variables = [x, y]  # Combine all variables - x, y and z
    for var in z:
        combined_variables.append(var)
    data = data[combined_variables].dropna()  # Drop missing values
    n = data.shape[0]  # Number of samples after dropping missing values
    k = data.shape[1] - 2  # Number of covariates
    assert n > 2

    if method == "spearman":
        V = data.rank(na_option="keep").cov()  # Change data to rank for spearman correlation
    else:
        V = data.astype(float).cov()  # Computing Covariance Matrix
    Vi = np.linalg.pinv(V, hermitian=True)  # Computing Inverse Covariance Matrix
    Vi_diag = Vi.diagonal()  # Storing variance
    D = np.diag(np.sqrt(1 / Vi_diag))  # Storing Standard Deviations from diagonal of inverse covariance matrix
    pcor = -1 * (D @ Vi @ D)
    r = pcor[0, 1]

    if np.isnan(r):
        return {"n": n, "r": np.nan, "CI95%": np.nan, "p-val": np.nan}

    # Finding p-value using student T test
    dof = n - k - 2  # Degree of freedom for multivariate analysis
    tval = r * np.sqrt(dof / (1 - r**2))  # Test statistic
    pval = 2 * t.sf(np.abs(tval), dof)  # Calculate p-value corresponding to the test statistic and degree of freedom

    ci = compute_ci(r=r, nx=(n - k), ny=(n - k))  # Finding Confidence Interval
    ci = np.round(ci, 3)
    stats = {
        "n": n,
        "r": r,
        "CI95%": [ci],
        "p-val": pval.round(5),
    }
    return stats


def entropy(x):
    """ "
    Returns entropy for a random variable x
    H(x) = - Σ p(x)log(p(x))
    :param x : random variable to calculate entropy for
    :returns : entropy of random variable
    """
    d = defaultdict(lambda: 0)
    s = 0.0
    entr = 0.0
    for i in x:
        d[i] += 1  # Calculating frequency of an event
        s += 1
    for i in d:
        p = d[i] / s  # Calculating probability for an event
        entr -= p * log(p, 2)  # H(x) = - Σ p(x)log(p(x))
    return entr


def conditional_MI(data=None, x=None, y=None, z=None):
    """
    Method to return conditional mutual information between X and Y given Z
    I(X, Y | Z) = H(X|Z) - H(X|Y,Z)
                = H(X,Z) - H(Z) - H(X,Y,Z) + H(Y,Z)
                = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    :param data : dataset
    :param x,y,z : column names from dataset
    :returns : conditional mutual information between X and Y given Z
    """
    X = data[list(x)].astype(int)
    Y = data[list(y)].astype(int)
    t = list(z)
    Z = data[t].astype(int)
    Z = Z.values.tolist()
    Z = list(data[t].itertuples(index=False, name=None))
    Hxz = entropy(map(lambda x: "%s/%s" % x, zip(X, Z)))  # Finding Joint entropy of X and Z
    Hyz = entropy(map(lambda x: "%s/%s" % x, zip(Y, Z)))  # Finding Joint entropy of Y and Z
    Hz = entropy(Z)  # Finding Entropy of Z
    Hxyz = entropy(map(lambda x: "%s/%s/%s" % x, zip(X, Y, Z)))  # Finding Joint Entropy of X, Y and Z
    return Hxz + Hyz - Hxyz - Hz
