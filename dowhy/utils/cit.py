from scipy.stats import t, norm
import pandas as pd
import numpy as np


def compute_ci(stat=None, nx=None, ny=None, confidence=.95):
    """Parametric confidence intervals around correlation coefficient.
    :param stat : correlation coefficient
    :param nx : length of vector x
    :param ny :length of vector y
    :param confidence : Confidence level (0.95 = 95%)
    
    :returns : array containing confidence interval
    """

    assert stat is not None and nx is not None
    assert isinstance(confidence, float)
    assert 0 < confidence < 1


    z = np.arctanh(stat) 
    se = 1 / np.sqrt(nx - 3)
    
    crit = np.abs(norm.ppf((1 - confidence) / 2))
    ci_z = np.array([z - crit * se, z + crit * se])
    ci = np.tanh(ci_z)  
    return np.round(ci, 5)

def partial_corr(data=None, x=None, y=None, z=None, method="pearson"):
    """ Calculate Partial correlation.
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

    assert data.shape[0] > 2
    assert x != z
    assert y != z
    assert x != y
    if isinstance(z, list):
        assert x not in z
        assert y not in z

    combined_variables = [x,y]
    for var in z:
        combined_variables.append(var)
    data = data[combined_variables].dropna()
    n = data.shape[0] 
    k = data.shape[1] - 2  
    assert n > 2

    if method == "spearman":
        V = data.rank(na_option='keep').cov()
    else:
        V = data.astype(float).cov()
    Vi = np.linalg.pinv(V, hermitian=True)  
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pcor = -1 * (D @ Vi @ D)  
    if z is not None:
        r = pcor[0, 1]
    else:
        with np.errstate(divide='ignore'):
            spcor = pcor / \
                np.sqrt(np.diag(V))[..., None] / \
                np.sqrt(np.abs(Vi_diag - Vi ** 2 / Vi_diag[..., None])).T
        r = spcor[1, 0] 
            

    if np.isnan(r):
        return {'n': n, 'r': np.nan, 'CI95%': np.nan, 'p-val': np.nan}

 
    dof = n - k - 2
    tval = r * np.sqrt(dof / (1 - r**2))
    pval = 2 * t.sf(np.abs(tval), dof)

    ci = compute_ci(stat=r, nx=(n - k), ny=(n - k))
    ci=np.round(ci, 3)
    stats = {
        'n': n,
        'r': r,
        'CI95%': [ci],
        'p-val': pval.round(5),
    }
    return stats
