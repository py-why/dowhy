import copy
import numpy as np
import pandas as pd
import logging

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter
from dowhy.causal_estimator import CausalEstimator

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

class DummyOutcomeRefuter(CausalRefuter):
    """Refute an estimate by replacing the outcome with a randomly generated variable.

    Supports additional parameters that can be specified in the refute_estimate() method.

    - 'dummy_outcome_type': str, None by default
    Default is to generate random values for the treatment. If palcebo_type is "permute",
    then the original treatement values are permuted by row.
    - 'num_simulations': int, CausalRefuter.DEFAULT_NUM_SIMULATIONS by default
    The number of simulations to be run
    - 'random_state': int, RandomState, None by default
    The seed value to be added if we wish to repeat the same random behavior. If we want to repeat the
    same behavior we push the same seed in the psuedo-random generator
    - 'outcome_function': function pd.Dataframe -> np.ndarray, None
    A function that takes in a function that takes the input data frame as the input and outputs the outcome
    variable. This allows us to create an output varable that only depends on the confounders and does not depend 
    on the treatment variable.
    Currently it supports some common functions like 
        1. Linear Regression
        2. K Nearest Neighbours
        3. Support Vector Machine
        4. Neural Network
        5. Random Forest
    - 'params': dict, default empty dictionary
    The parameters that go with the outcome_function. This consists of the paramters to be passed to the sklearn objects
    to give the desired behavior.
    - 'required_variables': int, list, bool, True by default
    A user can input either an integer value,list or bool.
        1. An integer argument refers to how many variables will be used for estimating the value of the outcome
        2. A list allows the user to explicitly refer to which variables will be used to estimate the outcome
            Furthermore, a user can either choose to select the variables desired. Or they can delselect the variables,
            that they do not want in their analysis. 
            For example:
            We need to pass required_variables = [W0,W1] is we want W0 and W1.
            We need to pass required_variables = [-W0,-W1] if we want all variables excluding W0 and W1.
    - scale: float, 1.0 by default
    The value by which the std_deviation of the dummy outcome has to be scaled 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dummy_outcome_type = kwargs.pop("placebo_type", None)
        if self._dummy_outcome_type is None:
            self._dummy_outcome_type = "Random Data"
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        self._random_state = kwargs.pop("random_state", None)
        self._outcome_function = kwargs.pop("outcome_function", None)
        self._params = kwargs.pop("params", None)
        self._scale = kwargs.pop("scale", 1)
        required_variables = kwargs.pop("required_variables", True)

        self._chosen_variables = self.choose_variables(required_variables)
        if self._params is None:
            self._params = {}

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        
        self.logger = logging.getLogger(__name__)

    def refute_estimate(self):

        # We need to change the identified estimand
        # We thus, make a copy. This is done as we don't want
        # to change the original DataFrame
        identified_estimand = copy.deepcopy(self._target_estimand)
        identified_estimand.outcome_variable = ["dummy_outcome"]

        sample_estimates = np.zeros(self._num_simulations)
        self.logger.info("Refutation over {} simulated datasets of {} treatment"
                        .format(self._num_simulations
                        ,self._dummy_outcome_type)
                        )
        num_rows =  self._data.shape[0]
        new_outcome = np.zeros((num_rows,))

        if self._outcome_function is not None:
            if callable(self._outcome_function):
                new_outcome = self._outcome_function(self._data)
            else:
                new_outcome = self._run_std_function()

            if type(new_outcome) is pd.Series or \
                type(new_outcome) is pd.DataFrame:
                new_outcome = new_outcome.values
            
            # Check if data types match
            assert type(new_outcome) is np.ndarray, ("Only  supports numpy.ndarray as the output")
            assert 'float' in new_outcome.dtype.name, ("Only float outcomes are currently supported")
            
            self._scale *= np.std(new_outcome)

            if len(new_outcome.shape) == 2 and \
                ( new_outcome.shape[0] ==1 or new_outcome.shape[1] ):
                self.logger.warning("Converting the row or column vector to 1D array")
                new_outcome = new_outcome.ravel()
                assert len(new_outcome) == num_rows, ("The number of outputs do not match that of the number of outcomes")
            elif len(new_outcome.shape) == 1:
                assert len(new_outcome) == num_rows, ("The number of outputs do not match that of the number of outcomes")
            else:
                raise Exception("Type Mismatch: The outcome is one dimensional, but the output has the shape:{}".format(new_outcome.shape))

        for index in range(self._num_simulations):

            if self._dummy_outcome_type == "permute":
                if self._random_state is None:
                    new_outcome = self._data[self._outcome_name].sample(frac=1).values
                else:
                    new_outcome = self._data[self._outcome_name].sample(frac=1,
                                                                random_state=self._random_state).values
            else:
                new_outcome += np.random.randn(num_rows)

        # Create a new column in the data by the name of dummy_outcome
        new_data = self._data.assign(dummy_outcome=new_outcome)

        # Sanity check the data
        self.logger.debug(new_data[0:10])

        new_estimator = CausalEstimator.get_estimator_object(new_data, identified_estimand, self._estimate)
        new_effect = new_estimator.estimate_effect()
        sample_estimates[index] = new_effect.value
        print(sample_estimates)

        refute = CausalRefutation(self._estimate.value,
                                        np.mean(sample_estimates),
                                        refutation_type="Refute: Use a Dummy Outcome")
        
        # Note: We hardcode the estimate value to ZERO as we want to check if it falls in the distribution of the refuter
        # Ideally we should expect that ZERO should fall in the distribution of the effect estimates as we have severed any causal 
        # relationship between the treatment and the outcome.

        dummy_estimator = copy.deepcopy(self._estimate)
        dummy_estimator.value = 0

        refute.add_significance_test_results(
            self.test_significance(dummy_estimator, sample_estimates)
        )

        return refute

    def _run_std_function(self):
        estimator = self._get_regressor_object()
        X = self._data[self._chosen_variables]
        y = self._data['y']
        estimator = estimator.fit(X, y)
        
        return estimator.predict(X) 
    
    def _get_regressor_object(self):
        
        if  self._outcome_function == "linear_regression":
            return LinearRegression(**self._params)
        elif self._outcome_function == "knn":
            return KNeighborsRegressor(**self._params)
        elif self._outcome_function == "svm":
            return SVR(**self._params)
        elif self._outcome_function == "random_forest":
            return RandomForestRegressor(**self._params)
        elif self._outcome_function == "neural_network":
            return MLPRegressor(**self._params)
        else:
            raise ValueError("The function: {} is not supported by dowhy at the moment")
