from typing import Any, List, Optional, Union

import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class PropensityScoreStratificationEstimator(PropensityScoreEstimator):
    """Estimate effect of treatment by stratifying the data into bins with
    identical common causes.

    Straightforward application of the back-door criterion.

    Supports additional parameters as listed below.

    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        num_strata: Union[str, int] = "auto",
        clipping_threshold: int = 10,
        propensity_score_model: Optional[Any] = None,
        propensity_score_column: str = "propensity_score",
        **kwargs,
    ):
        """
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param test_significance: Binary flag or a string indicating whether to test significance and by which method. All estimators support test_significance="bootstrap" that estimates a p-value for the obtained estimate using the bootstrap method. Individual estimators can override this to support custom testing methods. The bootstrap method supports an optional parameter, num_null_simulations. If False, no testing is done. If True, significance of the estimate is tested using the custom method if available, otherwise by bootstrap.
        :param evaluate_effect_strength: (Experimental) whether to evaluate the strength of effect
        :param confidence_intervals: Binary flag or a string indicating whether the confidence intervals should be computed and which method should be used. All methods support estimation of confidence intervals using the bootstrap method by using the parameter confidence_intervals="bootstrap". The bootstrap method takes in two arguments (num_simulations and sample_size_fraction) that can be optionally specified in the params dictionary. Estimators may also override this to implement their own confidence interval method. If this parameter is False, no confidence intervals are computed. If True, confidence intervals are computed by the estimator's specific method if available, otherwise through bootstrap
        :param num_null_simulations: The number of simulations for testing the
            statistical significance of the estimator
        :param num_simulations: The number of simulations for finding the
            confidence interval (and/or standard error) for a estimate
        :param sample_size_fraction: The size of the sample for the bootstrap
            estimator
        :param confidence_level: The confidence level of the confidence
            interval estimate
        :param need_conditional_estimates: Boolean flag indicating whether
            conditional estimates should be computed. Defaults to True if
            there are effect modifiers in the graph
        :param num_quantiles_to_discretize_cont_cols: The number of quantiles
            into which a numeric effect modifier is split, to enable
            estimation of conditional treatment effect over it.
        :param num_strata: Number of bins by which data will be stratified.
            Default is automatically determined.
        :param clipping_threshold: Mininum number of treated or control units
            per strata. Default=10
        :param propensity_score_model: The model used to compute propensity
            score. Can be any classification model that supports fit() and
            predict_proba() methods. If None, use
            LogisticRegression model as the default.
        :param propensity_score_column: Column name that stores the propensity
        score. Default='propensity_score'
        :param kwargs: (optional) Additional estimator-specific parameters
        """
        super().__init__(
            identified_estimand=identified_estimand,
            test_significance=test_significance,
            evaluate_effect_strength=evaluate_effect_strength,
            confidence_intervals=confidence_intervals,
            num_null_simulations=num_null_simulations,
            num_simulations=num_simulations,
            sample_size_fraction=sample_size_fraction,
            confidence_level=confidence_level,
            need_conditional_estimates=need_conditional_estimates,
            num_quantiles_to_discretize_cont_cols=num_quantiles_to_discretize_cont_cols,
            propensity_score_model=propensity_score_model,
            propensity_score_column=propensity_score_column,
            num_strata=num_strata,
            clipping_threshold=clipping_threshold,
            **kwargs,
        )

        self.logger.info("Using Propensity Score Stratification Estimator")

        # setting method-specific parameters
        self.num_strata = num_strata
        self.clipping_threshold = clipping_threshold

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        super().fit(data, effect_modifier_names=effect_modifier_names)
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        return self

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        **_,
    ):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        if self.propensity_score_column not in data:
            self.estimate_propensity_score_column(data)

        clipped = None
        # Infer the right strata based on clipping threshold
        if self.num_strata == "auto":
            # 0.5 because there are two values for the treatment
            clipping_t = self.clipping_threshold
            num_strata = 0.5 * data.shape[0] / clipping_t
            # To be conservative and allow most strata to be included in the
            # analysis
            strata_found = False
            while not strata_found:
                self.logger.info("'num_strata' selected as {}".format(num_strata))
                try:
                    clipped = self._get_strata(
                        data,
                        num_strata,
                        self.clipping_threshold,
                    )
                    num_ret_strata = clipped.groupby(["strata"]).count().reset_index()
                    # At least 90% of the strata should be included in analysis
                    if num_ret_strata.shape[0] >= 0.5 * num_strata:
                        strata_found = True
                    else:
                        num_strata = int(num_strata / 2)
                        self.logger.info(
                            f"Less than half the strata have at least {self.clipping_threshold} data points. Selecting fewer number of strata."
                        )
                        if num_strata < 2:
                            raise ValueError(
                                "Not enough data to generate at least two strata. This error may be due to a high value of 'clipping_threshold'."
                            )
                except ValueError:
                    self.logger.info(
                        "No strata found with at least {} data points. Selecting fewer number of strata".format(
                            self.clipping_threshold
                        )
                    )
                    num_strata = int(num_strata / 2)
                    if num_strata < 2:
                        raise ValueError(
                            "Not enough data to generate at least two strata. This error may be due to a high value of 'clipping_threshold'."
                        )
        else:
            clipped = self._get_strata(
                data,
                self.num_strata,
                self.clipping_threshold,
            )

        # sum weighted outcomes over all strata  (weight by treated population)
        weighted_outcomes = clipped.groupby("strata").agg(
            {self._target_estimand.treatment_variable[0]: ["sum"], "dbar": ["sum"], "d_y": ["sum"], "dbar_y": ["sum"]}
        )
        weighted_outcomes.columns = ["_".join(x) for x in weighted_outcomes.columns.to_numpy().ravel()]
        treatment_sum_name = self._target_estimand.treatment_variable[0] + "_sum"
        control_sum_name = "dbar_sum"

        weighted_outcomes["d_y_mean"] = weighted_outcomes["d_y_sum"] / weighted_outcomes[treatment_sum_name]
        weighted_outcomes["dbar_y_mean"] = weighted_outcomes["dbar_y_sum"] / weighted_outcomes["dbar_sum"]
        weighted_outcomes["effect"] = weighted_outcomes["d_y_mean"] - weighted_outcomes["dbar_y_mean"]
        total_treatment_population = weighted_outcomes[treatment_sum_name].sum()
        total_control_population = weighted_outcomes[control_sum_name].sum()
        total_population = total_treatment_population + total_control_population
        self.logger.debug(
            "Total number of data points is {0}, including {1} from treatment and {2} from control.".format(
                total_population, total_treatment_population, total_control_population
            )
        )

        if target_units == "att":
            est = (
                weighted_outcomes["effect"] * weighted_outcomes[treatment_sum_name]
            ).sum() / total_treatment_population
        elif target_units == "atc":
            est = (weighted_outcomes["effect"] * weighted_outcomes[control_sum_name]).sum() / total_control_population
        elif target_units == "ate":
            est = (
                weighted_outcomes["effect"]
                * (weighted_outcomes[control_sum_name] + weighted_outcomes[treatment_sum_name])
            ).sum() / total_population
        else:
            raise ValueError("Target units string value not supported")

        # TODO - how can we add additional information into the returned estimate?
        #        such as how much clipping was done, or per-strata info for debugging?
        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=est,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            propensity_scores=data[self.propensity_score_column],
        )

        estimate.add_estimator(self)
        return estimate

    def _get_strata(self, data: pd.DataFrame, num_strata, clipping_threshold):
        # sort the dataframe by propensity score
        # create a column 'strata' for each element that marks what strata it belongs to
        num_rows = data[self._target_estimand.outcome_variable[0]].shape[0]
        data["strata"] = ((data[self.propensity_score_column].rank(ascending=True) / num_rows) * num_strata).round(0)
        # for each strata, count how many treated and control units there are
        # throw away strata that have insufficient treatment or control

        data["dbar"] = 1 - data[self._target_estimand.treatment_variable[0]]  # 1-Treatment
        data["d_y"] = (
            data[self._target_estimand.treatment_variable[0]] * data[self._target_estimand.outcome_variable[0]]
        )
        data["dbar_y"] = data["dbar"] * data[self._target_estimand.outcome_variable[0]]
        stratified = data.groupby("strata")
        clipped = stratified.filter(
            lambda strata: min(
                strata.loc[strata[self._target_estimand.treatment_variable[0]] == 1].shape[0],
                strata.loc[strata[self._target_estimand.treatment_variable[0]] == 0].shape[0],
            )
            > clipping_threshold
        )
        self.logger.debug(
            "After using clipping_threshold={0}, here are the number of data points in each strata:\n {1}".format(
                clipping_threshold,
                clipped.groupby(["strata", self._target_estimand.treatment_variable[0]])[
                    self._target_estimand.outcome_variable
                ].count(),
            )
        )
        if clipped.empty:
            raise ValueError(
                "Method requires strata with number of data points per treatment > clipping_threshold (={0}). No such strata exists. Consider decreasing 'num_strata' or 'clipping_threshold' parameters.".format(
                    clipping_threshold
                )
            )
        return clipped

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.get_adjustment_set()
        expr += "+".join(var_list)
        return expr
