#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:24:48 2018

@author: mgralle

Debugging script for the dowhy package, using the Lalonde data example.
"""

#To simplify debugging, I obtained the Lalonde data as described on the DoWhy
#page and wrote it to a CSV file:

#from rpy2.robjects import r as R
#%load_ext rpy2.ipython
##%R install.packages("Matching")
#%R library(Matching)
#%R data(lalonde)
#%R -o lalonde
#lfile("lalonde.csv","w")
#lalonde.to_csv(lfile,index=False)
#lfile.close()

import pandas as pd
lalonde=pd.read_csv("lalonde.csv")

print("Lalonde data frame:")
print(lalonde.describe())

from dowhy.do_why import CausalModel

# 1. Propensity score weighting
model=CausalModel(
        data = lalonde,
        treatment='treat',
        outcome='re78',
        common_causes='nodegr+black+hisp+age+educ+married'.split('+'))
identified_estimand = model.identify_effect()

psw_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_weighting")
print("\n(1) Causal Estimate from PS weighting is " + str(psw_estimate.value))

psw_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_weighting")
print("\n(2) Causal Estimate from PS weighting is " + str(psw_estimate.value))


#2. Propensity score matching
psm_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching")
print("\n(1) Causal estimate from PS matching is " + str(psm_estimate.value))

psm_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching")
print("\n(2) Causal estimate from PS matching is " + str(psm_estimate.value))

#3. Linear regression
linear_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True)
print("\n(1) Causal estimate from linear regression is " + str(linear_estimate.value))

linear_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True)
print("\n(2) Causal estimate from linear regression is " + str(linear_estimate.value))

# Recreate model from scratch for linear regression
model=CausalModel(
        data = lalonde,
        treatment='treat',
        outcome='re78',
        common_causes='nodegr+black+hisp+age+educ+married'.split('+'))

identified_estimand = model.identify_effect()

linear_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True)
print("\n(3) Causal estimate from linear regression is " + str(linear_estimate.value))

print("\nLalonde Data frame hasn't changed:")
print(lalonde.describe())