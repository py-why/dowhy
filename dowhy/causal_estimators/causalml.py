import inspect
import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from importlib import import_module
import causalml

class Causalml(CausalEstimator):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Add the identification method used in the estimator
        self.identifier_method = self._target_estimand.identifier_method
        self.logger.debug("The identifier method used {}".format(self.identifier_method))

        # Check the backdoor variables being used
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._target_estimand.get_backdoor_variables()))
        
        # Add the observed confounders and one hot encode the categorical variables
        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables()
        if self._observed_common_causes_names:
            # Get the data of the unobserved confounders
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            # One hot encode the data if they are categorical
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes = []
        
        # Check the instrumental variables involved
        self.logger.debug("Instrumental variables used:"+
                        ",".join(self._target_estimand.instrumental_variables))
        
        # Perform the same actions as the above
        self._instrumental_variable_names = self._target_estimand.instrumental_variables
        if self._instrumental_variable_names:
            self._instrumental_variables = self._data[self._instrumental_variable_names]
            self._instrumental_variables = pd.get_dummies(self._instrumental_variables, drop_first=True)
        else:
            self._instrumental_variables = []

        # Check if effect modifiers are used
        self.logger.debug("Effect Modifiers used:" + 
                        ",".join(self._effect_modifier_names))
        
        
        # Get the class corresponding the the estimator to be used
        estimator_class = self._get_causalml_class_object(self._causalml_methodname)
        # Initialize the object
        self.estimator = estimator_class(**self.method_params["init_params"])
        self.logger.info("INFO: Using CausalML Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)
        
    def _get_causalml_class_object(self, module_method_name, *args, **kwargs):
        
        try:
            (module_name, _, class_name) = module_method_name.rpartition(".")
            estimator_module = import_module(module_name)
            estimator_class = getattr(estimator_module, class_name)

        except (AttributeError, AssertionError, ImportError):
            raise ImportError('Error loading {}.{}. Double-check the method name and ensure that all econml dependencies are installed.'.format(module_name, class_name))
        return estimator_class

    def _estimate_effect(self):
        X_names = self._observed_common_causes_names + \
                self._effect_modifier_names
        
        # Both the outcome and the treatment have to be 1D arrays according to the CausalML API
        y_name = self._outcome_name 
        treatment_name = self._treatment_name[0] # As we have only one treatment variable
        # We want to pass 'v0' rather than ['v0'] to prevent a shape mismatch
        
        func_args={
            'X':self._data[X_names],
            'y':self._data[y_name],
            'treatment':self._data[treatment_name]
        }

        arg_names = inspect.getfullargspec(self.estimator.estimate_ate)[0]
        matched_args = {
            arg: func_args[arg] for arg in func_args.keys() if arg in arg_names 
        }
        print(matched_args)
        value_tuple = self.estimator.estimate_ate(**matched_args) 

        # For CATEs
        arg_names = inspect.getfullargspec(self.estimator.fit_predict)[0]
        matched_args = {
            arg: func_args[arg] for arg in func_args.keys() if arg in arg_names 
        }
        cate_estimates = self.estimator.fit_predict(**matched_args) 

        estimate = CausalEstimate(estimate=value_tuple[0],
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  cate_estimates = cate_estimates,
                                  effect_intervals=(value_tuple[1],value_tuple[2]),
                                  _estimator_object=self.estimator)

        return estimate
        

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO we are conditioning on a postive treatment
        # TODO create an expression corresponding to each estimator used
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
 
