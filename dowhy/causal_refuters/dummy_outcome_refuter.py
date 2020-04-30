'''
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP  
WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP WIP 
'''
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
    - 'transformations': list, [('zero',''),('noise','DummyOutcomeRefuter.DEFAULT_STD_DEV')]
    The transformations list gives a list of actions to be performed to obtain the outcome. The actions are of the following types:
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

    The transformations list is of the following form:
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
    # The default standard deviation for noise
    DEFAULT_STD_DEV = 0.1 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        self._transformations = kwargs.pop("transformations",[("zero",""),("noise", 1)])
        
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
        self.logger.info("The transformation passed: {}".format(self._transformations) )

        data_chunks = self.data_preprocess()
        estimates = []
        for chunk in data_chunks:

            X_input = chunk[self._chosen_variables]
            new_outcome = chunk['y']
            X = self._data[self._chosen_variables]

            for action, func_args in self._transformations:

                if callable(action):
                    estimator = action(X_input, new_outcome, **func_args)
                    new_outcome = estimator(X)
                elif action in DummyOutcomeRefuter.SUPPORTED_ESTIMATORS:
                    estimator = self._estimate_dummy_outcome(func_args, action, new_outcome, X_input)
                    new_outcome = estimator(X)
                elif action == 'noise':
                    new_outcome = self._noise(new_outcome, func_args)
                elif action == 'permute':
                    new_outcome = self._permute(new_outcome, func_args)
                elif action =='zero':
                    new_outcome = np.zeros(new_outcome.shape)

            new_data = chunk.assign(dummy_outcome=new_outcome)
            new_estimator = CausalEstimator.get_estimator_object(new_data, identified_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            estimates.append(new_effect.value)

        # # This flag is to make sure we store the estimators whose input is deterministic
        # save_estimators = True
        # # We store the value of the estimators in the format "estimator_name" +  "pos_in_transform" : estimator_object
        # saved_estimator_dict = {}
        
        # X = self._data[self._chosen_variables]
        # new_outcome = self._data['y']
        
        # for index in range(self._num_simulations):
        #     transform_num = 0
        #     for action, func_args in self._transformations:

        #         if callable(action):
        #             new_outcome = action(X, **func_args)

        #         elif action in DummyOutcomeRefuter.SUPPORTED_ESTIMATORS:
        #             if action + str(transform_num) in saved_estimator_dict:
        #                 estimator = saved_estimator_dict[action + str(transform_num)]
        #                 new_outcome = estimator(X)
        #             else:
        #                 estimator = self._estimate_dummy_outcome(func_args, action, new_outcome)
        #                 new_outcome = estimator(X)
        #                 if save_estimators:
        #                     saved_estimator_dict[action + str(transform_num)] = estimator

        #         elif action == 'noise':
        #             save_estimators = False
        #             new_outcome = self._noise(new_outcome, func_args)

        #         elif action == 'permute':
        #             save_estimators = False
        #             new_outcome = self._permute(new_outcome, func_args)

        #         elif action =='zero':
        #             save_estimators = False
        #             new_outcome = np.zeros(new_outcome.shape)
            
        #         transform_num += 1
            
        #     save_estimators = False       
        
        # Create a new column in the data by the name of dummy_outcome
        
        # new_data = self._data.assign(dummy_outcome=new_outcome)

        # # Sanity check the data
        # self.logger.debug(new_data[0:10])

        # new_estimator = CausalEstimator.get_estimator_object(new_data, identified_estimand, self._estimate)
        # new_effect = new_estimator.estimate_effect()
        # sample_estimates[index] = new_effect.value

        refute = CausalRefutation(self._estimate.value,
                                        np.mean(sample_estimates),
                                        refutation_type="Refute: Use a Dummy Outcome")
        
        # Note: We hardcode the estimate value to ZERO as we want to check if it falls in the distribution of the refuter
        # Ideally we should expect that ZERO should fall in the distribution of the effect estimates as we have severed any causal 
        # relationship between the treatment and the outcome.

        dummy_estimator = copy.deepcopy(self._estimate)
        dummy_estimator.value = 0

        # refute.add_significance_test_results(
        #     self.test_significance(dummy_estimator, sample_estimates)
        # )

        return refute

    def data_preprocess(self):
        data_chunks = []

        assert len(self._treatment_name) == 1, "At present, DoWhy supports a simgle treatment variable"
        
        treatment_variable_name = self._target_estimand.treatment_name[0] # As we only have a single treatment
        variable_type = self._data[treatment_variable_name].dtypes
        
        if bool == variable_type:
            # All the positive values go the first bucket
            data_chunks.append( self._data[ self._data[treatment_variable_name] ])
            # All the negative values go into the other
            data_chunks.append( ~self._data[ self._data[treatment_variable_name] ])

        # We use string arguments to account for both 32 and 64 bit varaibles
        elif 'float' in variable_type.name or\
               'int' in variable_type.name:
            # action for continuous variables
            data_copy = copy.deepcopy( self._data )
            data_copy['bins'] = pd.qcut(data_copy[treatment_variable_name], 10)
            groups = data_copy.groupby('bins')
            data_chunks = [groups.get_group(group) for group in groups ]

        elif 'categorical' in variable_type.name:
            # Action for categorical variables
            groups = data_copy.groupby(treatment_variable_name)
            data_chunks = [groups.get_group(group) for group in groups ]
        else:
            raise ValueError("Passed {}. Expected bool, float, int or categorical".format(variable_type.name))

        return data_chunks
            
    def _estimate_dummy_outcome(self, func_args, action, outcome, X_chunk):
        estimator = self._get_regressor_object(action, func_args)
        X = X_chunk
        y = outcome
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

    def _permute(self, new_outcome, permute_fraction):
        '''
        In this function, we make use of the Fisher Yates shuffle:
        https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        '''
        if permute_fraction == 1:
            new_outcome = pd.DataFrame(new_outcome)
            new_outcome.columns = ['y']
            return new_outcome['y'].sample(frac=1).values
        else:
            permute_fraction /= 2 # We do this as every swap leads to two changes 
            changes = np.where( np.random.uniform(0,1,new_outcome.shape[0]) <= permute_fraction )[0] # As this is tuple containing a single element (array[...])
            num_rows = new_outcome.shape[0]
            for change in changes:
                if change + 1 < num_rows:
                    index = np.random.randint(change+1,num_rows)
                    temp = new_outcome[change]
                    new_outcome[change] = new_outcome[index]
                    new_outcome[index] = temp
            return new_outcome

    def _noise(self, new_outcome, std_dev):
        return new_outcome + np.random.normal(scale=std_dev,size=new_outcome.shape[0])