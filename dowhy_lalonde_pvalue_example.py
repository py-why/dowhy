#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:24:48 2018

@author: mgralle
"""

import os, sys
#sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath(""))


import dowhy
from dowhy.do_why import CausalModel
from rpy2.robjects import r as R
#%load_ext rpy2.ipython

#%R install.packages("Matching")
#%R library(Matching)

#%R data(lalonde)
#%R -o lalonde

#%%
#model=CausalModel(
#        data = lalonde,
#        treatment='treat',
#        outcome='re78',
#        common_causes='nodegr+black+hisp+age+educ+married'.split('+'))
#identified_estimand = model.identify_effect()
#
#linear_estimate = model.estimate_effect(identified_estimand,
#        method_name="backdoor.linear_regression",
#        test_significance=True)
##print(estimate)
#print("Causal estimate from linear regression is " + str(linear_estimate.value))
#print("p-value is " + str(linear_estimate.significance_test['p_value']))

#%%
model=CausalModel(
        data = lalonde,
        treatment='treat',
        outcome='re78',
        common_causes='nodegr+black+hisp+age+educ+married'.split('+'))
identified_estimand = model.identify_effect()

psw_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        test_significance=True, method_params={'num_simulations':100})
#print(estimate)
print("Causal Estimate from PS weighting is " + str(psw_estimate.value))
print("p-value is " + str(psw_estimate.significance_test['p_value']))

#%%
model=CausalModel(
        data = lalonde,
        treatment='treat',
        outcome='re78',
        common_causes='nodegr+black+hisp+age+educ+married'.split('+'))
identified_estimand = model.identify_effect()

psm_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching",
        test_significance=True, method_params={'num_simulations':10})
#print(estimate)
print("Causal estimate from PS matching is " + str(psm_estimate.value))
print("p-value is " + str(psm_estimate.significance_test['p_value']))
