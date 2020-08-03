import copy
import logging
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter
from dowhy.causal_estimator import CausalEstimator

class AddUnobservedCommonCause(CausalRefuter):

    """Add an unobserved confounder for refutation.

    Supports additional parameters that can be specified in the refute_estimate() method.

    - 'confounders_effect_on_treatment': how the simulated confounder affects the value of treatment. This can be linear (for continuous treatment) or binary_flip (for binary treatment)
    - 'confounders_effect_on_outcome': how the simulated confounder affects the value of outcome. This can be linear (for continuous outcome) or binary_flip (for binary outcome)
    - 'effect_strength_on_treatment': parameter for the strength of the effect of simulated confounder on treatment. For linear effect, it is the regression coeffient. For binary_flip, it is the probability that simulated confounder's effect flips the value of treatment from 0 to 1 (or vice-versa).
    - 'effect_strength_on_outcome': parameter for the strength of the effect of simulated confounder on outcome. For linear effect, it is the regression coeffient. For binary_flip, it is the probability that simulated confounder's effect flips the value of outcome from 0 to 1 (or vice-versa).

    TODO: Needs scaled version of the parameters and an interpretation module
    (e.g., in comparison to biggest effect of known confounder)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the parameters required for the refuter

        :param effect_on_t: str : This is used to represent the type of effect on the treatment due to the unobserved confounder.
        :param effect_on_y: str : This is used to represent the type of effect on the outcome due to the unobserved confounder.
        :param kappa_t: float, numpy.ndarray: This refers to the strength of the confounder on treatment. For a linear effect, it behaves like the regression coeffecient. For a binary flip it is the probability with which it can invert the value of the treatment.
        :param kappa_y: floar, numpy.ndarray: This refers to the strength of the confounder on outcome. For a linear effect, it behaves like the regression coefficient. For a binary flip, it is the probability with which it can invert the value of the outcome.
        """
        super().__init__(*args, **kwargs)

        self.effect_on_t = kwargs["confounders_effect_on_treatment"] if "confounders_effect_on_treatment" in kwargs else "binary_flip"
        self.effect_on_y = kwargs["confounders_effect_on_outcome"] if "confounders_effect_on_outcome" in kwargs else "linear"
        self.kappa_t = kwargs["effect_strength_on_treatment"]
        self.kappa_y = kwargs["effect_strength_on_outcome"]

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def refute_estimate(self):
        """
        This function attempts to add an unobserved common cause to the outcome and the treatment. At present, we have implemented the behavior for one dimensional behaviors for continueous
        and binary variables. This function can either take single valued inputs or a range of inputs. The function then looks at the data type of the input and then decides on the course of
        action.

        :return: CausalRefuter: An object that contains the estimated effect and a new effect and the name of the refutation used.
        """
        if not isinstance(self.kappa_t, np.ndarray) and not isinstance(self.kappa_y, np.ndarray): # Deal with single value inputs
            new_data = copy.deepcopy(self._data)
            new_data = self.include_confounders_effect(new_data, self.kappa_t, self.kappa_y)

            new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            refute = CausalRefutation(self._estimate.value, new_effect.value,
                                    refutation_type="Refute: Add an Unobserved Common Cause")
            
            refute.new_effect = np.array(new_effect.value)
            refute.add_refuter(self)
            return refute

        else: # Deal with multiple value inputs
            
            if isinstance(self.kappa_t, np.ndarray) and isinstance(self.kappa_y, np.ndarray): # Deal with range inputs
                                
                # Get a 2D matrix of values
                x,y =  np.meshgrid(self.kappa_t, self.kappa_y) # x,y are both MxN
                
                results_matrix = np.random.rand(len(x),len(y)) # Matrix to hold all the results of NxM
                print(results_matrix.shape)
                orig_data = copy.deepcopy(self._data)
                
                for i in range(0,len(x[0])):
                    for j in range(0,len(y)):
                        new_data = self.include_confounders_effect(orig_data, x[0][i], y[j][0])
                        new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
                        new_effect = new_estimator.estimate_effect()
                        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                                refutation_type="Refute: Add an Unobserved Common Cause")
                        self.logger.debug(refute)
                        results_matrix[i][j] = refute.estimated_effect # Populate the results
                
                fig = plt.figure(figsize=(6,5))
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                ax = fig.add_axes([left, bottom, width, height]) 

                cp = plt.contourf(x, y, results_matrix)
                plt.colorbar(cp)
                ax.set_title('Effect of Unobserved Common Cause')
                ax.set_xlabel('Value of Linear Constant on Treatment')
                ax.set_ylabel('Value of Linear Constant on Outcome')
                plt.show()

                refute.new_effect = results_matrix
                # Store the values into the refute object
                refute.add_refuter(self)
                return refute

            elif isinstance(self.kappa_t, np.ndarray):
                outcomes = np.random.rand(len(self.kappa_t))
                orig_data = copy.deepcopy(self._data)

                for i in range(0,len(self.kappa_t)):
                    new_data = self.include_confounders_effect(orig_data, self.kappa_t[i], self.kappa_y)
                    new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
                    new_effect = new_estimator.estimate_effect()
                    refute = CausalRefutation(self._estimate.value, new_effect.value,
                                            refutation_type="Refute: Add an Unobserved Common Cause")
                    self.logger.debug(refute)
                    outcomes[i] = refute.estimated_effect # Populate the results

                fig = plt.figure(figsize=(6,5))
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                ax = fig.add_axes([left, bottom, width, height]) 

                plt.plot(self.kappa_t, outcomes)
                ax.set_title('Effect of Unobserved Common Cause')
                ax.set_xlabel('Value of Linear Constant on Treatment')
                ax.set_ylabel('New Effect')
                plt.show() 

                refute.new_effect = outcomes
                refute.add_refuter(self)
                return refute

            elif isinstance(self.kappa_y, np.ndarray):
                outcomes = np.random.rand(len(self.kappa_y))
                orig_data = copy.deepcopy(self._data)

                for i in range(0, len(self.kappa_y)):
                    new_data = self.include_confounders_effect(orig_data, self.kappa_t, self.kappa_y[i])
                    new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
                    new_effect = new_estimator.estimate_effect()
                    refute = CausalRefutation(self._estimate.value, new_effect.value,
                                            refutation_type="Refute: Add an Unobserved Common Cause")
                    self.logger.debug(refute)
                    outcomes[i] = refute.estimated_effect # Populate the results
                
                fig = plt.figure(figsize=(6,5))
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                ax = fig.add_axes([left, bottom, width, height])

                plt.plot(self.kappa_y, outcomes)
                ax.set_title('Effect of Unobserved Common Cause')
                ax.set_xlabel('Value of Linear Constant on Outcome')
                ax.set_ylabel('New Effect')
                plt.show() 

                refute.new_effect = outcomes
                refute.add_refuter(self)
                return refute

    def include_confounders_effect(self, new_data, kappa_t, kappa_y):
        """
        This function deals with the change in the value of the data due to the effect of the unobserved confounder. 
        In the case of a binary flip, we flip only if the random number is greater than the threshold set.
        In the case of a linear effect, we use the variable as the linear regression constant.

        :param new_data: pandas.DataFrame: The data to be changed due to the effects of the unobserved confounder.
        :param kappa_t: numpy.float64: The value of the threshold for binary_flip or the value of the regression coefficient for linear effect.
        :param kappa_y: numpy.float64: The value of the threshold for binary_flip or the value of the regression coefficient for linear effect.

        :return: pandas.DataFrame: The DataFrame that includes the effects of the unobserved confounder.
        """
        num_rows = self._data.shape[0]
        w_random=np.random.randn(num_rows)

        if self.effect_on_t == "binary_flip":
            new_data['temp_rand_no'] = np.random.random(num_rows)
            new_data.loc[new_data['temp_rand_no'] <= kappa_t, self._treatment_name ]  = 1- new_data.loc[new_data['temp_rand_no'] <= kappa_t, self._treatment_name]
            for tname in self._treatment_name:
                if pd.api.types.is_bool_dtype(self._data[tname]):
                    new_data = new_data.astype({tname: 'bool'}, copy=False)
            new_data.pop('temp_rand_no')
        elif self.effect_on_t == "linear":
            confounder_t_effect = kappa_t * w_random
            new_data[self._treatment_name] = new_data[self._treatment_name].values - np.ndarray(shape=(num_rows,1), buffer=confounder_t_effect)
        else:
            raise NotImplementedError("'" + self.effect_on_t + "' method not supported for confounders' effect on treatment")

        if self.effect_on_y == "binary_flip":
            new_data['temp_rand_no'] = np.random.random(num_rows)
            new_data.loc[new_data['temp_rand_no'] <= kappa_y, self._outcome_name ]  = 1- new_data[self._outcome_name]
            new_data.pop('temp_rand_no')
        elif self.effect_on_y == "linear":
            confounder_y_effect = kappa_y * w_random
            new_data[self._outcome_name] = new_data[self._outcome_name].values - np.ndarray(shape=(num_rows,1), buffer=confounder_y_effect)
        else:
            raise NotImplementedError("'" + self.effect_on_y+ "' method not supported for confounders' effect on outcome")
        return new_data
