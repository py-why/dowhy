import copy
import numpy as np
import pandas as pd
import logging
import pdb

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

    - 'num_simulations': int, CausalRefuter.DEFAULT_NUM_SIMULATIONS by default
    The number of simulations to be run
    - 'pipeline': list, [noise]
    The pipeline gives a list of actions to be performed to obtain the outcome. The actions are of the following types:
    * function argument: function pd.Dataframe -> np.ndarray
        It takes in a function that takes the input data frame as the input and outputs the outcome
        variable. This allows us to create an output varable that only depends on the covariates and does not depend 
        on the treatment variable.
    * string argument
        - Currently it supports some common functions like 
            1. Linear Regression
            2. K Nearest Neighbours
            3. Support Vector Machine
            4. Neural Network
            5. Random Forest
        - On the other hand, there are other options:
            1. Permute
            This permutes the rows of the outcome, disassociating any effect of the treatment on the outcome.
            2. Noise
            This adds white noise to the outcome with white noise, reducing any causal relationship with the treatment.
            3. Zero
            It replaces all the values in the outcome by zero

    The pipeline is of the following form:
    * If the function pd.Dataframe -> np.ndarray is already defined.
    [(func,func_params),('permute', permute_fraction), ('noise', std_dev)]
    * If a function from the above list is used
    [('knn',{'n_neighbors':5}), ('permute', permute_fraction), ('noise', std_dev)]

    - 'required_variables': int, list, bool, True by default
    The inputs are either an integer value, list or bool.
        1. An integer argument refers to how many variables will be used for estimating the value of the outcome
        2. A list explicitly refers to which variables will be used to estimate the outcome
            Furthermore, it gives the ability to explictly select or deselect the covariates present in the estimation of the 
            outcome. This is done by either adding or explicitly removing variables from the list as shown below: 
            For example:
            We need to pass required_variables = [W0,W1] is we want W0 and W1.
            We need to pass required_variables = [-W0,-W1] if we want all variables excluding W0 and W1.
        3. If the value is True, we wish to include all variables to estimate the value of the outcome. A False value is INVALID
           and will result in an error. 
    Note:
    These inputs are fed to the function for estimating the outcome variable. The same set of required_variables is used for each
    instance of an internal function.
    """
    # The currently supported estimators
    SUPPORTED_ESTIMATORS = ["linear_regression", "knn", "svm", "random_forest", "neural_network"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        pipeline = kwargs.pop("pipeline",[("zero",""),("noise", 1)])
        pipeline = self._parse_pipeline(pipeline)
        self._pipeline = self._save_precompute(pipeline)
        pdb.set_trace()
        required_variables = kwargs.pop("required_variables", True)

        if required_variables is False:
            raise ValueError("The value of required_variables cannot be False")

        self._chosen_variables = self.choose_variables(required_variables)

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
        self.logger.info("Refutation over {} simulated datasets".format(self._num_simulations) )

        new_outcome = self._data['y']

        # if callable(self._outcome_function):
        #     new_outcome = self._outcome_function(self._data)
        

        # elif type(self._outcome_function) is str:
        #     new_outcome = self._estimate_dummy_outcome()

        # if type(new_outcome) is pd.Series or \
        #     type(new_outcome) is pd.DataFrame:
        #     new_outcome = new_outcome.values
        
        # # Check if data types match
        # assert type(new_outcome) is np.ndarray, ("Only  supports numpy.ndarray as the output")
        # assert 'float' in new_outcome.dtype.name, ("Only float outcomes are currently supported")
        
        # self._scale *= np.std(new_outcome)

        # if len(new_outcome.shape) == 2 and \
        #     ( new_outcome.shape[0] ==1 or new_outcome.shape[1] ):
        #     self.logger.warning("Converting the row or column vector to 1D array")
        #     new_outcome = new_outcome.ravel()
        #     assert len(new_outcome) == num_rows, ("The number of outputs do not match that of the number of outcomes")
        # elif len(new_outcome.shape) == 1:
        #     assert len(new_outcome) == num_rows, ("The number of outputs do not match that of the number of outcomes")
        # else:
        #     raise Exception("Type Mismatch: The outcome is one dimensional, but the output has the shape:{}".format(new_outcome.shape))

        
        pdb.set_trace()
        for index in range(self._num_simulations):
            for action, props in self._pipeline:
                if props['input'] == ['X']:
                    new_outcome = action( self._data[self._chosen_variables] )
                elif props['input'] == ['y']:
                    new_outcome = action( new_outcome )
                elif props['input'] == ['X','y']:
                    temp_estimator = action( new_outcome )
                    new_outcome = temp_estimator( self._data[self._chosen_variables] )   

        # Create a new column in the data by the name of dummy_outcome
        
        new_data = self._data.assign(dummy_outcome=new_outcome)

        # Sanity check the data
        self.logger.debug(new_data[0:10])

        new_estimator = CausalEstimator.get_estimator_object(new_data, identified_estimand, self._estimate)
        new_effect = new_estimator.estimate_effect()
        sample_estimates[index] = new_effect.value

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
            
    def _estimate_dummy_outcome(self, func_args, action, new_data):
        pdb.set_trace()
        estimator = self._get_regressor_object(action, func_args)
        X = self._data[self._chosen_variables]
        y = new_data
        estimator = estimator.fit(X, y)
        
        return estimator.predict
    
    def _get_regressor_object(self, action, func_args):
        if  action == "linear_regression":
            return LinearRegression(**func_args)
        elif action == "knn":
            return KNeighborsRegressor(**func_args)
        elif action == "svm":
            return SVR(**func_args)
        elif action == "random_forest":
            return RandomForestRegressor(**func_args)
        elif action == "neural_network":
            return MLPRegressor(**func_args)
        else:
            raise ValueError("The function: {} is not supported by dowhy at the moment".format(action))

    def _permute(self, new_data, permute_fraction):
        '''
        In this function, we make use of the Fisher Yates shuffle:
        https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        '''
        if permute_fraction == 1:
            new_data = pd.DataFrame(new_data)
            new_data.columns = ['y']
            return new_data['y'].sample(frac=1).values
        else: 
            changes = np.where( np.random.uniform(0,1,new_data.shape[0]) <= permute_fraction )[0]
            num_rows = new_data.shape[0]
            for change in changes:
                index = np.random.randint(change+1,num_rows)
                temp = new_data[change]
                new_data[change] = new_data[index]
                new_data[index] = temp

            return new_data

    def _noise(self, new_data, std_dev):
        return new_data + np.random.randn(new_data.shape[0]) * std_dev