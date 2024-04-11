import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator
from dowhy.interpreters.visual_interpreter import VisualInterpreter


class ConfounderDistributionInterpreter(VisualInterpreter):
    SUPPORTED_ESTIMATORS = [
        PropensityScoreWeightingEstimator,
    ]

    def __init__(self, estimate, fig_size, font_size, var_name, var_type, **kwargs):
        """
        :param estimate: Causal estimate
        :param fig_size: Size of the figure
        :param font_size: Size of the font of the plot title
        :param var_name: The confounding variable for which distribution changes should be compared
        :param var_type: Type of the confounding variable; must be one of 'continuous' or 'discrete'
        """

        super().__init__(estimate, **kwargs)
        if not isinstance(estimate, CausalEstimate):
            error_msg = "The interpreter method expects a CausalEstimate object."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.estimator = self.estimate.estimator
        if not any(
            isinstance(self.estimator, est_class)
            for est_class in ConfounderDistributionInterpreter.SUPPORTED_ESTIMATORS
        ):
            error_msg = "The interpreter method only supports propensity score weighting estimator."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if var_type not in {"continuous", "discrete"}:
            error_msg = "var_type must be one of 'continuous' or 'discrete'."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if var_type == "continuous":
            error_msg = "Distributional changes plot for continuous variables is not yet implemented."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.fig_size = fig_size
        self.font_size = font_size
        self.var_name = var_name

    @staticmethod
    def discrete_dist_plot(labels, not_treated_counts, treated_counts, ax, title, var_name, font_size, width=0.35):
        """
        Plot of the treated vs untreated.
        """

        ax.bar(labels - width / 2, not_treated_counts, width, label="Untreated")
        ax.bar(labels + width / 2, treated_counts, width, label="Treated")
        ax.set_xlabel(var_name)
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=font_size)
        ax.set_xticks(labels)
        ax.set_xticklabels(labels)
        ax.legend()

    def interpret(self, data: pd.DataFrame):
        """
        Shows distribution changes for confounding variables before and after applying inverse propensity weights.
        """

        cols = self.estimator._observed_common_causes_names + self.estimate._treatment_name
        df = data[cols].copy()
        treated = self.estimate._treatment_name[0]
        propensity = self.estimate.propensity_scores

        # add weight column
        df.loc[:, "weight"] = df.loc[:, treated] * (propensity) ** (-1) + (1 - df.loc[:, treated]) * (
            1 - propensity
        ) ** (-1)

        # before weights are applied we count number rows in each category
        # which is equivalent to summing over weight=1
        barplot_df_before = df.groupby([self.var_name, treated]).size().reset_index(name="count")

        # after weights are applied we need to sum over the given weights
        barplot_df_after = df.groupby([self.var_name, treated]).agg({"weight": np.sum}).reset_index()
        barplot_df_after.rename(columns={"weight": "count"}, inplace=True)

        title1 = "Distribution of " + self.var_name + " before applying the weights"
        title2 = "Distribution of " + self.var_name + " after applying the weights"

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size)
        iterable = zip([barplot_df_before, barplot_df_after], [ax1, ax2], [title1, title2])
        for plot_df, ax, title in iterable:
            aggregated_not_treated = plot_df[plot_df[treated] == False].reset_index()
            aggregated_treated = plot_df[plot_df[treated] == True].reset_index()

            labels = aggregated_not_treated[self.var_name].astype("float")
            not_treated_counts = aggregated_not_treated["count"]

            treated_counts = aggregated_treated["count"]
            self.discrete_dist_plot(
                labels, not_treated_counts, treated_counts, ax, title, self.var_name, self.font_size
            )

        fig.tight_layout()
        plt.show()
