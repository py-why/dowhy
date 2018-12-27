#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:24:48 2018

@author: mgralle
"""

import os, sys
sys.path.append(os.path.abspath("../../"))

import dowhy
from dowhy.do_why import CausalModel
from rpy2.robjects import r as R
%load_ext rpy2.ipython

#%R install.packages("Matching")
%R library(Matching)

%R data(lalonde)
%R -o lalonde

model=CausalModel(
        data = lalonde,
        treatment='treat',
        outcome='re78',
        common_causes='nodegr+black+hisp+age+educ+married'.split('+'))
identified_estimand = model.identify_effect()

linear_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        test_significance=1)
#print(estimate)
print("Causal Estimate is " + str(linear_estimate.value))
print("p-value " + str(linear_estimate.significance_test))

psw_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_weighting")
#print(estimate)
print("Causal Estimate is " + str(psw_estimate.value))