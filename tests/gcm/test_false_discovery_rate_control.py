#from dowhy.gcm.FalseDiscoveryRateControl import numberfy, adjusted_FDR_pvalues
from false_discovery_rate_control import numberfy, adjusted_fdr_pvalues, adjusted_fdcr_pvalues
import pytest
import numpy as np
import math
from scipy.stats import rankdata

def test_numberfy():
    assert numberfy(17.1)   == 17.1
    assert numberfy("3.7")  == 3.7
    assert (math.isnan(numberfy("x.1")))
    assert (math.isnan(numberfy(np.nan)))

def test_adjFDR():
    adjFDR=adjusted_fdr_pvalues((0.05, 0.01, 0.1))
    assert abs(adjFDR[0] - 0.075)< 0.0001
    assert abs(adjFDR[1] - 0.030)< 0.0001
    assert abs(adjFDR[2] - 0.100)< 0.0001

    adjFDR=adjusted_fdr_pvalues([0.01, "-", 0.1])
    assert abs(adjFDR[0] - 0.020)< 0.0001
    assert (math.isnan(adjFDR[1]))
    assert abs(adjFDR[2] - 0.100)< 0.0001

    adjFDR=adjusted_fdr_pvalues((0.01, 0.022, 0.023, 0.05,0.07,0.1))
    assert abs(adjFDR[0] - 0.046)< 0.0001
    assert abs(adjFDR[1] - 0.046)< 0.0001
    assert abs(adjFDR[2] - 0.046)< 0.0001
    assert abs(adjFDR[3] - 0.075)< 0.0001
    assert abs(adjFDR[4] - 0.084)< 0.0001
    assert abs(adjFDR[5] - 0.100)< 0.0001

    adjFDR=adjusted_fdr_pvalues((0.01, np.nan , 0.023, "-",0.07,0.1))
    # [0.04              nan 0.046             nan 0.09333333 0.1       ]
    assert abs(adjFDR[0] - 0.040)< 0.0001
    assert (math.isnan(adjFDR[1]))
    assert abs(adjFDR[2] - 0.046)< 0.0001
    assert (math.isnan(adjFDR[3]))
    assert abs(adjFDR[4] - 0.09333333)< 0.00000001
    assert abs(adjFDR[5] - 0.100)< 0.0001

def test_adjusted_fdcr_pvalues()
    #missmatched lengh pvalue vec and belife score vec
    adjFDCR = adjusted_fdcr_pvalues((0.05, 0.01, 0.1),(1,1),0)
    assert adjFDCR == -17

    adjFDCR=adjusted_fdcr_pvalues((0.05, 0.01, 0.1),(1,1,1),0)
    #[0.0500005  0.0100002  0.1        0.00333343]
    assert abs(adjFDCR[0] - 0.050) < 0.0001
    assert abs(adjFDCR[1] - 0.010) < 0.0001
    assert abs(adjFDCR[2] - 0.100) < 0.0001
    assert abs(adjFDCR[3] - 0.0033)< 0.0001

    adjFDCR=adjusted_fdcr_pvalues((0.05, 0.01, 0.1),(0,0,1),1)
    #[0.05000025 0.02       0.1        0.02      ]
    assert abs(adjFDCR[0] - 0.050) < 0.0001
    assert abs(adjFDCR[1] - 0.020) < 0.0001
    assert abs(adjFDCR[2] - 0.100) < 0.0001
    assert abs(adjFDCR[3] - 0.020) < 0.0001

    adjFDCR=adjusted_fdcr_pvalues((0.01, np.nan , 0.023, "-",0.07,0.1),(1,1,1,1,1,1),0.5)
    #[0.02000007        nan 0.03450006        nan 0.08400006 0.1  0.00750005]
    assert abs(adjFDCR[0] - 0.020)  < 0.0001
    assert (math.isnan(adjFDCR[1]))
    assert abs(adjFDCR[2] - 0.0345) < 0.0001
    assert (math.isnan(adjFDCR[3]))
    assert abs(adjFDCR[4] - 0.084)  < 0.0001
    assert abs(adjFDCR[5] - 0.100)  < 0.0001
    assert abs(adjFDCR[6] - 0.0075) < 0.0001

    adjFDCR=adjusted_fdcr_pvalues((0.1, 0.023, 0.07, 0.01),(1,1,1,1),0)
    #[0.1        0.02300046 0.0700007  0.0100003  0.0025001 ]
    assert abs(adjFDCR[0] - 0.100)  < 0.0001
    assert abs(adjFDCR[1] - 0.023) < 0.0001
    assert abs(adjFDCR[2] - 0.070) < 0.0001
    assert abs(adjFDCR[3] - 0.010) < 0.0001
    assert abs(adjFDCR[4] - 0.0025)  < 0.0001    
