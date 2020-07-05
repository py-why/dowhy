import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from dowhy.interpreters.visual_interpreter import VisualInterpreter
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator
from dowhy.causal_estimator import CausalEstimate


class ConfounderDistributionIntepreter(VisualInterpreter):
    SUPPORTED_ESTIMATORS = [PropensityScoreWeightingEstimator, ]

    def __init__(self, estimate, **kwargs):
        super().__init__(estimate, **kwargs)
        if not isinstance(estimate, CausalEstimate):
            error_msg = "The interpreter method expects a CausalEstimate object."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.estimator = self.estimate.estimator
        if not any(isinstance(self.estimator, est_class) for est_class in ConfounderDistributionIntepreter.SUPPORTED_ESTIMATORS):
            error_msg = "The interpreter method only supports propensity score weighting estimator."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def interpret(self, fig_size=(10, 7), font_size=14):

        """
        Shows distribution changes for categorical variables before and after applying inverse propensity weights.
        """

        fig_size = self.fig_size
        font_size = self.font_size
        var_name = self.var_name

        cols = self.estimator._observed_common_causes_names + self.estimator._treatment_name
        df = self.estimator._data[cols]
        treated = self.estimator._treatment_name[0]
        propensity = self.estimate.propensity_scores

        # add weight column
        df["weight"] = df[treated] * (propensity) ** (-1) + (1 - df[treated]) * (1 - propensity) ** (-1)

        # before weights are applied we count number rows in each category
        # which is equivalent to summing over weight=1
        barplot_df_before = df.groupby([var_name, treated]).size().reset_index(name="count")

        # after weights are applied we need to sum over the given weights
        barplot_df_after = df.groupby([var_name, treated]).agg({'weight': np.sum}).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        sns.barplot(x=var_name, y='count', data=barplot_df_before, orient='v', hue=treated, ax=ax1)
        ax1.set_title("Distribution of " + var_name + " before applying the weights", fontsize=font_size)

        sns.barplot(x=var_name, y='weight', data=barplot_df_after, orient='v', hue=treated, ax=ax2)
        ax2.set_title("Distribution of " + var_name + " after applying the weights", fontsize=font_size)

        fig.tight_layout()
        plt.show()
