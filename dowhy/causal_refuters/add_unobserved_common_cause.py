import copy
import logging
import numpy as np
import pandas as pd
import scipy.stats

import math
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

    TODO: Needs an interpretation module
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the parameters required for the refuter.

        If effect_strength_on_treatment or effect_strength_on_outcome is not
        given, it is calculated automatically as a range between the
        minimum and maximum effect strength of observed confounders on treatment
        and outcome respectively.

        :param confounders_effect_on_treatment: str : The type of effect on the treatment due to the unobserved confounder. Possible values are ['binary_flip', 'linear']
        :param confounders_effect_on_outcome: str : The type of effect on the outcome due to the unobserved confounder. Possible values are ['binary_flip', 'linear']
        :param effect_strength_on_treatment: float, numpy.ndarray: This refers to the strength of the confounder on treatment. For a linear effect, it behaves like the regression coeffecient. For a binary flip it is the probability with which it can invert the value of the treatment.
        :param effect_strength_on_outcome: float, numpy.ndarray: This refers to the strength of the confounder on outcome. For a linear effect, it behaves like the regression coefficient. For a binary flip, it is the probability with which it can invert the value of the outcome.
        :param effect_fraction_on_treatment: float: If effect_strength_on_treatment is not provided, this parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on treatment. Defaults to 1.
        :param effect_fraction_on_outcome: float: If effect_strength_on_outcome is not provided, this parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on outcome. Defaults to 1.
        :param plotmethod: string: Type of plot to be shown. If None, no plot is generated. This parameter is used only only when more than one treatment confounder effect values or outcome confounder effect values are provided. Default is "colormesh". Supported values are "contour", "colormesh" when more than one value is provided for both confounder effect value parameters; "line" when provided for only one of them.
        """
        super().__init__(*args, **kwargs)

        self.effect_on_t = kwargs["confounders_effect_on_treatment"] if "confounders_effect_on_treatment" in kwargs else "binary_flip"
        self.effect_on_y = kwargs["confounders_effect_on_outcome"] if "confounders_effect_on_outcome" in kwargs else "linear"
        self.kappa_t = kwargs["effect_strength_on_treatment"] if "effect_strength_on_treatment" in kwargs else None
        self.kappa_y = kwargs["effect_strength_on_outcome"] if "effect_strength_on_outcome" in kwargs else None
        self.frac_strength_treatment = kwargs["effect_fraction_on_treatment"] if "effect_fraction_on_treatment" in kwargs else 1
        self.frac_strength_outcome = kwargs["effect_fraction_on_outcome"] if "effect_fraction_on_outcome" in kwargs else 1
        self.simulated_method_name = kwargs["simulated_method_name"] if "simulated_method_name" in kwargs else "linear_based"
        self.plotmethod = kwargs['plotmethod'] if "plotmethod" in kwargs else "colormesh"
        self.logger = logging.getLogger(__name__)


    def infer_default_kappa_t(self, len_kappa_t = 10):
        """ Infer default effect strength of simulated confounder on treatment.

        """
        observed_common_causes_names = self._target_estimand.get_backdoor_variables()
        if len(observed_common_causes_names)>0:
            observed_common_causes = self._data[observed_common_causes_names]
            observed_common_causes = pd.get_dummies(observed_common_causes, drop_first=True)
        else:
            raise ValueError("There needs to be at least one common cause to" +
                    "automatically compute the default value of kappa_t."+
                " Provide a value for kappa_t")
        t = self._data[self._treatment_name]
        # Standardizing the data
        observed_common_causes = StandardScaler().fit_transform(observed_common_causes)
        if self.effect_on_t == "binary_flip":
            # Fit a model containing all confounders and compare predictions
            # using all features compared to all features except a given
            # confounder.
            tmodel = LogisticRegression().fit(observed_common_causes, t)
            tpred = tmodel.predict(observed_common_causes).astype(int)
            flips = []
            for i in  range(observed_common_causes.shape[1]):
                oldval = np.copy(observed_common_causes[:, i])
                observed_common_causes[:,i] = 0
                tcap = tmodel.predict(observed_common_causes).astype(int)
                observed_common_causes[:,i] = oldval
                flips.append(np.sum(abs(tcap-tpred))/tpred.shape[0])
            min_coeff, max_coeff = min(flips), max(flips)
        elif self.effect_on_t == "linear":
            # Estimating the regression coefficient from standardized features to t
            corrcoef_var_t = np.corrcoef(observed_common_causes, t, rowvar=False)[-1, :-1]
            std_dev_t = np.std(t)[0]
            max_coeff = max(corrcoef_var_t) * std_dev_t
            min_coeff = min(corrcoef_var_t) * std_dev_t
        else:
            raise NotImplementedError("'" + self.effect_on_t +
                    "' method not supported for confounders' effect on treatment")

        min_coeff, max_coeff = self._compute_min_max_coeff(min_coeff, max_coeff,
                self.frac_strength_treatment)
        # By default, return a plot with 10 points
        # consider 10 values of the effect of the unobserved confounder
        step = (max_coeff - min_coeff)/len_kappa_t
        self.logger.info("(Min, Max) kappa_t for observed common causes, ({0}, {1})".format(
            min_coeff, max_coeff))
        if np.equal(max_coeff, min_coeff):
            return max_coeff
        else:
            return np.arange(min_coeff, max_coeff, step)

    def _compute_min_max_coeff(self,
            min_coeff, max_coeff, effect_strength_fraction):
        max_coeff = effect_strength_fraction * max_coeff
        min_coeff = effect_strength_fraction * min_coeff
        return min_coeff, max_coeff

    def infer_default_kappa_y(self, len_kappa_y = 10):
        """ Infer default effect strength of simulated confounder on treatment.

        """
        observed_common_causes_names = self._target_estimand.get_backdoor_variables()
        if len(observed_common_causes_names)>0:
            observed_common_causes = self._data[observed_common_causes_names]
            observed_common_causes = pd.get_dummies(observed_common_causes, drop_first=True)
        else:
            raise ValueError("There needs to be at least one common cause to" +
                    "automatically compute the default value of kappa_y."+
                " Provide a value for kappa_y")
        y = self._data[self._outcome_name]
        # Standardizing the data
        observed_common_causes = StandardScaler().fit_transform(observed_common_causes)
        if self.effect_on_y == "binary_flip":
            # Fit a model containing all confounders and compare predictions
            # using all features compared to all features except a given
            # confounder.
            ymodel = LogisticRegression().fit(observed_common_causes, y)
            ypred = ymodel.predict(observed_common_causes).astype(int)
            flips = []
            for i in  range(observed_common_causes.shape[1]):
                oldval = np.copy(observed_common_causes[:, i])
                observed_common_causes[:,i] = 0
                ycap = ymodel.predict(observed_common_causes).astype(int)
                observed_common_causes[:,i] = oldval
                flips.append(np.sum(abs(ycap-ypred))/ypred.shape[0])
            min_coeff, max_coeff = min(flips), max(flips)
        elif self.effect_on_y == "linear":
            corrcoef_var_y = np.corrcoef(observed_common_causes, y, rowvar=False)[-1, :-1]
            std_dev_y = np.std(y)[0]
            max_coeff = max(corrcoef_var_y) * std_dev_y
            min_coeff = min(corrcoef_var_y) * std_dev_y
        else:
            raise NotImplementedError("'" + self.effect_on_y +
                    "' method not supported for confounders' effect on outcome")
        min_coeff, max_coeff = self._compute_min_max_coeff(min_coeff, max_coeff,
                self.frac_strength_outcome)
        # By default, return a plot with 10 points
        # consider 10 values of the effect of the unobserved confounder
        step = (max_coeff - min_coeff)/len_kappa_y
        self.logger.info("(Min, Max) kappa_y for observed common causes, ({0}, {1})".format(
            min_coeff, max_coeff))
        if np.equal(max_coeff, min_coeff):
            return max_coeff
        else:
            return np.arange(min_coeff, max_coeff, step)

    def refute_estimate(self):
        """
        This function attempts to add an unobserved common cause to the outcome and the treatment. At present, we have implemented the behavior for one dimensional behaviors for continuous
        and binary variables. This function can either take single valued inputs or a range of inputs. The function then looks at the data type of the input and then decides on the course of
        action.

        :return: CausalRefuter: An object that contains the estimated effect and a new effect and the name of the refutation used.
        """
        if self.kappa_t is None:
            self.kappa_t = self.infer_default_kappa_t()
        if self.kappa_y is None:
            self.kappa_y = self.infer_default_kappa_y()
        if not isinstance(self.kappa_t, (list, np.ndarray)) and not isinstance(self.kappa_y, (list,np.ndarray)): # Deal with single value inputs
            new_data = copy.deepcopy(self._data)
            new_data = self.include_confounders_effect(new_data, self.kappa_t, self.kappa_y)
            new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            refute = CausalRefutation(self._estimate.value, new_effect.value,
                                    refutation_type="Refute: Add an Unobserved Common Cause")

            refute.new_effect_array = np.array(new_effect.value)
            refute.new_effect = new_effect.value
            refute.add_refuter(self)
            return refute

        else: # Deal with multiple value inputs

            if isinstance(self.kappa_t, (list, np.ndarray)) and isinstance(self.kappa_y, (list, np.ndarray)): # Deal with range inputs
                # Get a 2D matrix of values
                #x,y =  np.meshgrid(self.kappa_t, self.kappa_y) # x,y are both MxN

                results_matrix = np.random.rand(len(self.kappa_t),len(self.kappa_y)) # Matrix to hold all the results of NxM
                orig_data = copy.deepcopy(self._data)
                for i in range(len(self.kappa_t)):
                    for j in range(len(self.kappa_y)):
                        new_data = self.include_confounders_effect(orig_data, self.kappa_t[i], self.kappa_y[j])
                        new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
                        new_effect = new_estimator.estimate_effect()
                        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                                refutation_type="Refute: Add an Unobserved Common Cause")
                        results_matrix[i][j] = refute.new_effect # Populate the results

                refute.new_effect_array = results_matrix
                refute.new_effect = (np.min(results_matrix), np.max(results_matrix))
                # Store the values into the refute object
                refute.add_refuter(self)
                if self.plotmethod is None:
                    return refute

                import matplotlib
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(6,5))
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                ax = fig.add_axes([left, bottom, width, height])

                oe = self._estimate.value
                contour_levels = [oe/4.0, oe/2.0, (3.0/4)*oe, oe]
                contour_levels.extend([0, np.min(results_matrix), np.max(results_matrix)])
                if self.plotmethod=="contour":
                    cp = plt.contourf(self.kappa_y, self.kappa_t, results_matrix,
                            levels=sorted(contour_levels))
                    # Adding a label on the contour line for the original estimate
                    fmt = {}
                    trueeffect_index = np.where(cp.levels==oe)[0][0]
                    fmt[cp.levels[trueeffect_index]] = "Estimated Effect"
                    # Label every other level using strings
                    plt.clabel(cp,[cp.levels[trueeffect_index]],inline=True, fmt=fmt)
                    plt.colorbar(cp)
                elif self.plotmethod=="colormesh":
                    cp = plt.pcolormesh(self.kappa_y, self.kappa_t, results_matrix,
                            shading="nearest")
                    plt.colorbar(cp, ticks=contour_levels)
                ax.yaxis.set_ticks(self.kappa_t)
                ax.xaxis.set_ticks(self.kappa_y)
                plt.xticks(rotation=45)
                ax.set_title('Effect of Unobserved Common Cause')
                ax.set_ylabel('Value of Linear Constant on Treatment')
                ax.set_xlabel('Value of Linear Constant on Outcome')
                plt.show()

                return refute

            elif isinstance(self.kappa_t, (list, np.ndarray)):
                outcomes = np.random.rand(len(self.kappa_t))
                orig_data = copy.deepcopy(self._data)

                for i in range(0,len(self.kappa_t)):
                    new_data = self.include_confounders_effect(orig_data, self.kappa_t[i], self.kappa_y)
                    new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
                    new_effect = new_estimator.estimate_effect()
                    refute = CausalRefutation(self._estimate.value, new_effect.value,
                                            refutation_type="Refute: Add an Unobserved Common Cause")
                    self.logger.debug(refute)
                    outcomes[i] = refute.new_effect # Populate the results

                refute.new_effect_array = outcomes
                refute.new_effect = (np.min(outcomes), np.max(outcomes))
                refute.add_refuter(self)
                if self.plotmethod is None:
                    return refute

                import matplotlib
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(6,5))
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                ax = fig.add_axes([left, bottom, width, height])

                plt.plot(self.kappa_t, outcomes)
                plt.axhline(self._estimate.value, linestyle='--',color="gray")
                ax.set_title('Effect of Unobserved Common Cause')
                ax.set_xlabel('Value of Linear Constant on Treatment')
                ax.set_ylabel('Estimated Effect after adding the common cause')
                plt.show()

                return refute

            elif isinstance(self.kappa_y, (list, np.ndarray)):
                outcomes = np.random.rand(len(self.kappa_y))
                orig_data = copy.deepcopy(self._data)

                for i in range(0, len(self.kappa_y)):
                    new_data = self.include_confounders_effect(orig_data, self.kappa_t, self.kappa_y[i])
                    new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
                    new_effect = new_estimator.estimate_effect()
                    refute = CausalRefutation(self._estimate.value, new_effect.value,
                                            refutation_type="Refute: Add an Unobserved Common Cause")
                    self.logger.debug(refute)
                    outcomes[i] = refute.new_effect # Populate the results

                refute.new_effect_array = outcomes
                refute.new_effect = (np.min(outcomes), np.max(outcomes))
                refute.add_refuter(self)
                if self.plotmethod is None:
                    return refute

                import matplotlib
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(6,5))
                left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
                ax = fig.add_axes([left, bottom, width, height])

                plt.plot(self.kappa_y, outcomes)
                plt.axhline(self._estimate.value, linestyle='--',color="gray")
                ax.set_title('Effect of Unobserved Common Cause')
                ax.set_xlabel('Value of Linear Constant on Outcome')
                ax.set_ylabel('Estimated Effect after adding the common cause')
                plt.show()

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
        stdnorm = scipy.stats.norm()
        w_random = stdnorm.rvs(num_rows)

        if self.effect_on_t == "binary_flip":
            alpha = 2*kappa_t-1 if kappa_t >=0.5 else 1-2*kappa_t
            interval = stdnorm.interval(alpha)
            rel_interval = interval[0] if kappa_t >=0.5 else interval[1]
            new_data.loc[rel_interval <= w_random, self._treatment_name ]  = 1- new_data.loc[rel_interval <= w_random, self._treatment_name]
            for tname in self._treatment_name:
                if pd.api.types.is_bool_dtype(self._data[tname]):
                    new_data = new_data.astype({tname: 'bool'}, copy=False)
        elif self.effect_on_t == "linear":
            confounder_t_effect = kappa_t * w_random
            # By default, we add the effect of simulated confounder for treatment.
            # But subtract it from outcome to create a negative correlation
            # assuming that the original confounder's effect was positive on both.
            # This is to remove the effect of the original confounder.
            new_data[self._treatment_name] = new_data[self._treatment_name].values + np.ndarray(shape=(num_rows,1), buffer=confounder_t_effect)
        else:
            raise NotImplementedError("'" + self.effect_on_t + "' method not supported for confounders' effect on treatment")

        if self.effect_on_y == "binary_flip":
            alpha = 2*kappa_y-1 if kappa_y >=0.5 else 1-2*kappa_y
            interval = stdnorm.interval(alpha)
            rel_interval = interval[0] if kappa_y >=0.5 else interval[1]
            new_data.loc[rel_interval <= w_random, self._outcome_name ]  = 1- new_data.loc[rel_interval <= w_random, self._outcome_name]
            for yname in self._outcome_name:
                if pd.api.types.is_bool_dtype(self._data[yname]):
                    new_data = new_data.astype({yname: 'bool'}, copy=False)
        elif self.effect_on_y == "linear":
            confounder_y_effect = (-1) * kappa_y * w_random
            # By default, we add the effect of simulated confounder for treatment.
            # But subtract it from outcome to create a negative correlation
            # assuming that the original confounder's effect was positive on both.
            # This is to remove the effect of the original confounder.
            new_data[self._outcome_name] = new_data[self._outcome_name].values + np.ndarray(shape=(num_rows,1), buffer=confounder_y_effect)
        else:
            raise NotImplementedError("'" + self.effect_on_y+ "' method not supported for confounders' effect on outcome")
        return new_data



    def include_simulated_confounder(self, convergence_threshold = 0.1, c_star_max = 1000):
        '''
        This function simulates an unobserved confounder based on the data using the following steps:
            1. It calculates the "residuals"  from the treatment and outcome model
                i.) The outcome model has outcome as the dependent variable and all the observed variables including treatment as independent variables
                ii.) The treatment model has treatment as the dependent variable and all the observed variables as independent variables.

            2. U is an intermediate random variable drawn from the normal distribution with the weighted average of residuals as mean and a unit variance
               U ~ N(c1*d_y + c2*d_t, 1)
               where
                *d_y and d_t are residuals from the treatment and outcome model
                *c1 and c2 are coefficients to the residuals

            3. The final U, which is the simulated unobserved confounder is obtained by debiasing the intermediate variable U by residualising it with X


        Choosing the coefficients c1 and c2:
        The coefficients are chosen based on these basic assumptions:
            1. There is a hyperbolic relationship satisfying c1*c2 = c_star
            2. c_star is chosen from a range of possible values based on the correlation of the obtained simulated variable with outcome and treatment.
            3. The product of correlations with treatment and outcome should be at a minimum distance to the maximum correlations with treatment and outcome in any of the observed confounders
            4. The ratio of the weights should be such that they maintain the ratio of the maximum possible observed coefficients within some confidence interval

        :param c_star_max: The maximum possible value for the hyperbolic curve on which the coefficients to the residuals lie. It defaults to 1000 in the code if not specified by the user.
        :type int
        :param convergence_threshold: The threshold to check the plateauing of the correlation while selecting a c_star. It defaults to 0.1 in the code if not specified by the user
        :type float
        :returns final_U: The simulated values of the unobserved confounder based on the data
        :type pandas.core.series.Series

        '''


        #Obtaining the list of observed variables
        required_variables = True
        observed_variables = self.choose_variables(required_variables)

        observed_variables_with_treatment_and_outcome = observed_variables + self._treatment_name + self._outcome_name

        #Taking a subset of the dataframe that has only observed variables
        self._data = self._data[observed_variables_with_treatment_and_outcome]

        #Residuals from the outcome model obtained by fitting a linear model
        y = self._data[self._outcome_name[0]]
        observed_variables_with_treatment = observed_variables + self._treatment_name
        X = self._data[observed_variables_with_treatment]
        model = sm.OLS(y,X.astype('float'))
        results = model.fit()
        residuals_y = y - results.fittedvalues
        d_y = list(pd.Series(residuals_y))


        #Residuals from the treatment model obtained by fitting a linear model
        t = self._data[self._treatment_name[0]].astype('int64')
        X = self._data[observed_variables]
        model = sm.OLS(t,X)
        results = model.fit()
        residuals_t = t - results.fittedvalues
        d_t = list(pd.Series(residuals_t))


        #Initialising product_cor_metric_observed with a really low value as finding maximum
        product_cor_metric_observed = -10000000000

        for i in observed_variables:
            current_obs_confounder = self._data[i]
            outcome_values = self._data[self._outcome_name[0]]
            correlation_y = current_obs_confounder.corr(outcome_values)
            treatment_values = t
            correlation_t = current_obs_confounder.corr(treatment_values)
            product_cor_metric_current = correlation_y*correlation_t
            if product_cor_metric_current>=product_cor_metric_observed:
                product_cor_metric_observed = product_cor_metric_current
                correlation_t_observed = correlation_t
                correlation_y_observed = correlation_y


        #The user has an option to give the the effect_strength_on_y and effect_strength_on_t which can be then used instead of maximum correlation with treatment and outcome in the observed variables as it specifies the desired effect.
        if self.kappa_t is not None:
            correlation_t_observed = self.kappa_t
        if self.kappa_y is not None:
            correlation_y_observed = self.kappa_y


        #Choosing a c_star based on the data.
        #The correlations stop increasing upon increasing c_star after a certain value, that is it plateaus and we choose the value of c_star to be the value it plateaus.

        correlation_y_list = []
        correlation_t_list = []
        product_cor_metric_simulated_list = []
        x_list = []


        step = int(c_star_max/10)
        for i in range(0, int(c_star_max), step):
            c1 = math.sqrt(i)
            c2 = c1
            final_U = self.generate_confounder_from_residuals(c1, c2, d_y, d_t, X)
            current_simulated_confounder = final_U
            outcome_values = self._data[self._outcome_name[0]]
            correlation_y = current_simulated_confounder.corr(outcome_values)
            correlation_y_list.append(correlation_y)

            treatment_values = t
            correlation_t = current_simulated_confounder.corr(treatment_values)
            correlation_t_list.append(correlation_t)

            product_cor_metric_simulated = correlation_y*correlation_t
            product_cor_metric_simulated_list.append(product_cor_metric_simulated)


            x_list.append(i)


        index = 1
        while index<len(correlation_y_list):
            if (correlation_y_list[index]-correlation_y_list[index-1])<=convergence_threshold:
                c_star = x_list[index]
                break
            index = index+1

        #Choosing c1 and c2 based on the hyperbolic relationship once c_star is chosen by going over various combinations of c1 and c2 values and choosing the combination which
        #which maintains the minimum distance between the product of correlations of the simulated variable and the product of maximum correlations of one of the observed variables
        # and additionally checks if the ratio of the weights are such that they maintain the ratio of the maximum possible observed coefficients within some confidence interval


        #c1_final and c2_final are initialised to the values on the hyperbolic curve such that c1_final = c2_final  and c1_final*c2_final = c_star
        c1_final = math.sqrt(c_star)
        c2_final = math.sqrt(c_star)


        #initialising min_distance_between_product_cor_metrics to be a value greater than 1
        min_distance_between_product_cor_metrics = 1.5
        i = 0.05

        threshold = c_star/0.05

        while i<=threshold:
            c2 = i
            c1 = c_star/c2
            final_U = self.generate_confounder_from_residuals(c1, c2, d_y, d_t, X)

            current_simulated_confounder = final_U
            outcome_values = self._data[self._outcome_name[0]]
            correlation_y = current_simulated_confounder.corr(outcome_values)

            treatment_values = t
            correlation_t = current_simulated_confounder.corr(treatment_values)

            product_cor_metric_simulated = correlation_y*correlation_t

            if min_distance_between_product_cor_metrics>=abs(product_cor_metric_simulated - product_cor_metric_observed):
                min_distance_between_product_cor_metrics = abs(product_cor_metric_simulated - product_cor_metric_observed)
                additional_condition = (correlation_y_observed/correlation_t_observed)
                if ((c1/c2) <= (additional_condition + 0.3*additional_condition)) and ((c1/c2) >= (additional_condition - 0.3*additional_condition)): #choose minimum positive value
                    c1_final = c1
                    c2_final = c2

            i = i*1.5

        '''#closed form solution

        print("c_star_max before closed form", c_star_max)

        if max_correlation_with_t == -1000:
            c2 = 0
            c1 = c_star_max
        else:
            additional_condition = abs(max_correlation_with_y/max_correlation_with_t)
            print("additional_condition", additional_condition)
            c2 = math.sqrt(c_star_max/additional_condition)
            c1 = c_star_max/c2'''

        final_U = self.generate_confounder_from_residuals(c1_final, c2_final, d_y, d_t, X)

        return final_U


    def generate_confounder_from_residuals(self, c1, c2, d_y, d_t, X):
        '''
        This function takes the residuals from the treatment and outcome model and their coefficients and simulates the intermediate random variable U by taking
        the row wise normal distribution corresponding to each residual value and then debiasing the intermediate variable to get the final variable

        :param c1: coefficient to the residual from the outcome model
        :type float
        :param c2: coefficient to the residual from the treatment model
        :type float
        :param d_y: residuals from the outcome model
        :type list
        :param d_t: residuals from the treatment model
        :type list

        :returns final_U: The simulated values of the unobserved confounder based on the data
        :type pandas.core.series.Series

        '''
        U = []

        for j in range(len(d_t)):
            simulated_variable_mean = c1*d_y[j]+c2*d_t[j]
            simulated_variable_stddev = 1
            U.append(np.random.normal(simulated_variable_mean, simulated_variable_stddev, 1))

        U = np.array(U)
        model = sm.OLS(U,X)
        results = model.fit()
        U = U.reshape(-1, )
        final_U = U - results.fittedvalues.values
        final_U = pd.Series(U)

        return final_U









