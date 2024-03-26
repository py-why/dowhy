import copy
from typing import Any, List, Optional, Type, Union

import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_identifier import EstimandType, IdentifiedEstimand
from dowhy.utils.api import parse_state


class TwoStageRegressionEstimator(CausalEstimator):
    """Compute treatment effect whenever the effect is fully mediated by
    another variable (front-door) or when there is an instrument available.

    Currently only supports a linear model for the effects.

    Supports additional parameters as listed below.

    """

    # First stage statistical model
    DEFAULT_FIRST_STAGE_MODEL = LinearRegressionEstimator
    # Second stage statistical model
    DEFAULT_SECOND_STAGE_MODEL = LinearRegressionEstimator

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
        first_stage_model: Optional[Union[CausalEstimator, Type[CausalEstimator]]] = None,
        second_stage_model: Optional[Union[CausalEstimator, Type[CausalEstimator]]] = None,
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
        :param first_stage_model: First stage estimator to be used. Default is
            linear regression.
        :param second_stage_model: Second stage estimator to be used. Default
            is linear regression.
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
            first_stage_model=first_stage_model,
            second_stage_model=second_stage_model,
            **kwargs,
        )
        self.logger.info("INFO: Using Two Stage Regression Estimator")
        # Check if the treatment is one-dimensional
        if len(self._target_estimand.treatment_variable) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)
        modified_target_estimand = copy.deepcopy(self._target_estimand)
        modified_target_estimand.identifier_method = "backdoor"
        modified_target_estimand.backdoor_variables = self._target_estimand.mediation_first_stage_confounders
        if first_stage_model is not None:
            self._first_stage_model = (
                first_stage_model
                if isinstance(first_stage_model, CausalEstimator)
                else first_stage_model(
                    modified_target_estimand,
                    test_significance=self._significance_test,
                    evaluate_effect_strength=self._effect_strength_eval,
                    confidence_intervals=self._confidence_intervals,
                    **kwargs,
                )
            )
        else:
            self._first_stage_model = self.__class__.DEFAULT_FIRST_STAGE_MODEL(
                modified_target_estimand,
                test_significance=self._significance_test,
                evaluate_effect_strength=self._effect_strength_eval,
                confidence_intervals=self._confidence_intervals,
                **kwargs,
            )
            self.logger.warning("First stage model not provided. Defaulting to sklearn.linear_model.LinearRegression.")

        modified_target_estimand = copy.deepcopy(self._target_estimand)
        modified_target_estimand.identifier_method = "backdoor"
        modified_target_estimand.backdoor_variables = self._target_estimand.mediation_second_stage_confounders
        if second_stage_model is not None:
            self._second_stage_model = (
                second_stage_model
                if isinstance(second_stage_model, CausalEstimator)
                else second_stage_model(
                    modified_target_estimand,
                    test_significance=self._significance_test,
                    evaluate_effect_strength=self._effect_strength_eval,
                    confidence_intervals=self._confidence_intervals,
                    **kwargs,
                )
            )
        else:
            self._second_stage_model = self.__class__.DEFAULT_SECOND_STAGE_MODEL(
                modified_target_estimand,
                test_significance=self._significance_test,
                evaluate_effect_strength=self._effect_strength_eval,
                confidence_intervals=self._confidence_intervals,
                **kwargs,
            )
            self.logger.warning("Second stage model not provided. Defaulting to backdoor.linear_regression.")

        if self._target_estimand.estimand_type == EstimandType.NONPARAMETRIC_NDE:
            modified_target_estimand.identifier_method = "backdoor"
            modified_target_estimand = copy.deepcopy(self._target_estimand)
            self._second_stage_model_nde = type(self._second_stage_model)(
                modified_target_estimand,
                test_significance=self._significance_test,
                evaluate_effect_strength=self._effect_strength_eval,
                confidence_intervals=self._confidence_intervals,
                **kwargs,
            )

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
        **_,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param iv_instrument_name: Name of the specific instrumental variable
            to be used. Needs to be one of the IVs identified in the
            identification step. Default is to use all the IV variables
            from the identification step.
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        self.reset_encoders()  # Forget any existing encoders
        self._set_effect_modifiers(data, effect_modifier_names)

        if len(self._target_estimand.treatment_variable) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)

        if self._target_estimand.identifier_method == "frontdoor":
            self.logger.debug("Front-door variable used:" + ",".join(self._target_estimand.get_frontdoor_variables()))
            self._frontdoor_variables_names = self._target_estimand.get_frontdoor_variables()
            if self._frontdoor_variables_names:
                if len(self._frontdoor_variables_names) > 1:
                    raise ValueError(
                        "Only singleton frontdoor variables are supported for estimation using TwoStageRegression."
                    )
                self._frontdoor_variables = data[self._frontdoor_variables_names]
            else:
                self._frontdoor_variables = None
                error_msg = "No front-door variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        elif self._target_estimand.identifier_method == "mediation":
            self.logger.debug("Mediators used:" + ",".join(self._target_estimand.get_mediator_variables()))
            self._mediators_names = self._target_estimand.get_mediator_variables()

            if self._mediators_names:
                if len(self._mediators_names) > 1:
                    raise ValueError(
                        "Only singleton mediator variables are supported for estimation using TwoStageRegression."
                    )
                self._mediators = data[self._mediators_names]
            else:
                self._mediators = None
                error_msg = "No mediator variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        elif self._target_estimand.identifier_method == "iv":
            self.logger.debug(
                "Instrumental variables used:" + ",".join(self._target_estimand.get_instrumental_variables())
            )
            self._instrumental_variables_names = self._target_estimand.get_instrumental_variables()

            if self._instrumental_variables_names:
                self._instrumental_variables = data[self._instrumental_variables_names]
            else:
                self._instrumental_variables = None
                error_msg = "No instrumental variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)

        if self._target_estimand.identifier_method == "frontdoor":
            self._first_stage_model._target_estimand.outcome_variable = parse_state(self._frontdoor_variables_names)
        elif self._target_estimand.identifier_method == "mediation":
            self._first_stage_model._target_estimand.outcome_variable = parse_state(self._mediators_names)

        self._first_stage_model.fit(
            data,
            effect_modifier_names=effect_modifier_names,
        )

        if self._target_estimand.identifier_method == "frontdoor":
            self._second_stage_model._target_estimand.treatment_variable = parse_state(self._frontdoor_variables_names)
        elif self._target_estimand.identifier_method == "mediation":
            self._second_stage_model._target_estimand.treatment_variable = parse_state(self._mediators_names)

        self._second_stage_model.fit(
            data,
            effect_modifier_names=effect_modifier_names,
        )

        if self._target_estimand.estimand_type == EstimandType.NONPARAMETRIC_NDE:
            self._second_stage_model_nde._target_estimand.identifier_method = "backdoor"
            self._second_stage_model_nde.fit(
                data,
                effect_modifier_names=effect_modifier_names,
            )

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

        estimate_value = None
        # First stage
        first_stage_estimate = self._first_stage_model.estimate_effect(
            data,
            control_value=control_value,
            treatment_value=treatment_value,
            target_units=target_units,
        )

        # Second Stage
        second_stage_estimate = self._second_stage_model.estimate_effect(
            data,
            control_value=control_value,
            treatment_value=treatment_value,
            target_units=target_units,
        )
        # Combining the two estimates
        natural_indirect_effect = first_stage_estimate.value * second_stage_estimate.value
        # This same estimate is valid for frontdoor as well as mediation (NIE)
        estimate_value = natural_indirect_effect
        self.symbolic_estimator = self.construct_symbolic_estimator(
            first_stage_estimate.realized_estimand_expr,
            second_stage_estimate.realized_estimand_expr,
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
        )
        if self._target_estimand.estimand_type == EstimandType.NONPARAMETRIC_NDE:

            total_effect_estimate = self._second_stage_model_nde.estimate_effect(
                data,
                control_value=control_value,
                treatment_value=treatment_value,
                target_units=target_units,
            )
            natural_direct_effect = total_effect_estimate.value - natural_indirect_effect
            estimate_value = natural_direct_effect
            self.symbolic_estimator = self.construct_symbolic_estimator(
                first_stage_estimate.realized_estimand_expr,
                second_stage_estimate.realized_estimand_expr,
                total_effect_estimate.realized_estimand_expr,
                estimand_type=self._target_estimand.estimand_type,
            )
        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=estimate_value,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
        )

        estimate.add_estimator(self)
        return estimate

    def build_first_stage_features(self, data_df: pd.DataFrame):
        treatment_vals = data_df[self._target_estimand.treatment_variable]
        if len(self._observed_common_causes_names) > 0:
            observed_common_causes_vals = data_df[self._observed_common_causes_names]
            observed_common_causes_vals = self._encode(observed_common_causes_vals, "observed_common_causes")

        if self._effect_modifier_names:
            effect_modifiers_vals = data_df[self._effect_modifier_names]
            effect_modifiers_vals = self._encode(effect_modifiers_vals, "effect_modifiers")

        if type(treatment_vals) is not np.ndarray:
            treatment_vals = treatment_vals.to_numpy()
        if treatment_vals.shape[0] != data_df.shape[0]:
            raise ValueError("Provided treatment values and dataframe should have the same length.")
        # Bulding the feature matrix
        n_samples = treatment_vals.shape[0]
        self.logger.debug("Number of samples" + str(n_samples) + str(len(self._target_estimand.treatment_variable)))
        treatment_2d = treatment_vals.reshape((n_samples, len(self._target_estimand.treatment_variable)))
        if len(self._observed_common_causes_names) > 0:
            features = np.concatenate((treatment_2d, observed_common_causes_vals), axis=1)
        else:
            features = treatment_2d
        if self._effect_modifier_names:
            for i in range(treatment_2d.shape[1]):
                curr_treatment = treatment_2d[:, i]
                new_features = curr_treatment[:, np.newaxis] * effect_modifiers_vals.to_numpy()
                features = np.concatenate((features, new_features), axis=1)
        features = features.astype(
            float, copy=False
        )  # converting to float in case of binary treatment and no other variables
        # features = sm.add_constant(features, has_constant='add') # to add an intercept term
        return features

    def construct_symbolic_estimator(
        self, first_stage_symbolic, second_stage_symbolic, total_effect_symbolic=None, estimand_type=None
    ):
        nie_symbolic = "(" + first_stage_symbolic + ")*(" + second_stage_symbolic + ")"
        if estimand_type == EstimandType.NONPARAMETRIC_NIE:
            return nie_symbolic
        elif estimand_type == EstimandType.NONPARAMETRIC_NDE:
            return "(" + total_effect_symbolic + ") - (" + nie_symbolic + ")"
