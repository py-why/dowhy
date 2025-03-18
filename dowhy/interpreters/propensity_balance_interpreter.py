import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_stratification_estimator import PropensityScoreStratificationEstimator
from dowhy.interpreters.visual_interpreter import VisualInterpreter


class PropensityBalanceInterpreter(VisualInterpreter):
    SUPPORTED_ESTIMATORS = [
        PropensityScoreStratificationEstimator,
    ]

    def __init__(self, estimate, **kwargs):
        super().__init__(estimate, **kwargs)
        if not isinstance(estimate, CausalEstimate):
            error_msg = "The interpreter method expects a CausalEstimate object."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.estimator = self.estimate.estimator
        if not any(
            isinstance(self.estimator, est_class) for est_class in PropensityBalanceInterpreter.SUPPORTED_ESTIMATORS
        ):
            error_msg = "The interpreter method only supports propensity score stratification estimator."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def interpret(self, data: pd.DataFrame):
        """Balance plot that shows the change in standardized mean differences for each covariate after propensity score stratification."""
        cols = (
            self.estimator._observed_common_causes_names
            + self.estimate._treatment_name
            + ["strata", "propensity_score"]
        )
        df = data[cols]
        df_long = (
            pd.wide_to_long(df.reset_index(), stubnames=["W"], i="index", j="common_cause_id")
            .reset_index()
            .astype({"W": "float64"})
        )

        # First, calculating mean differences by strata
        mean_diff = df_long.groupby(self.estimate._treatment_name + ["common_cause_id", "strata"]).agg(
            mean_w=("W", np.mean)
        )
        mean_diff = (
            mean_diff.groupby(["common_cause_id", "strata"]).transform(lambda x: x.max() - x.min()).reset_index()
        )
        mean_diff = mean_diff.query("v0==True")
        size_by_w_strata = (
            df_long.groupby(["common_cause_id", "strata"]).agg(size=("propensity_score", np.size)).reset_index()
        )
        size_by_strata = df_long.groupby(["common_cause_id"]).agg(size=("propensity_score", np.size)).reset_index()
        size_by_strata = pd.merge(size_by_w_strata, size_by_strata, on="common_cause_id")
        mean_diff_strata = pd.merge(mean_diff, size_by_strata, on=("common_cause_id", "strata"))

        stddev_by_w_strata = df_long.groupby(["common_cause_id", "strata"]).agg(stddev=("W", np.std)).reset_index()
        mean_diff_strata = pd.merge(mean_diff_strata, stddev_by_w_strata, on=["common_cause_id", "strata"])
        mean_diff_strata["scaled_mean"] = (mean_diff_strata["mean_w"] / mean_diff_strata["stddev"]) * (
            mean_diff_strata["size_x"] / mean_diff_strata["size_y"]
        )
        mean_diff_strata = (
            mean_diff_strata.groupby("common_cause_id").agg(std_mean_diff=("scaled_mean", np.sum)).reset_index()
        )

        # Second, without strata
        mean_diff_overall = df_long.groupby(self.estimate._treatment_name + ["common_cause_id"]).agg(
            mean_w=("W", np.mean)
        )
        mean_diff_overall = (
            mean_diff_overall.groupby("common_cause_id").transform(lambda x: x.max() - x.min()).reset_index()
        )
        mean_diff_overall = mean_diff_overall[mean_diff_overall[self.estimate._treatment_name[0]] == True]  # TODO
        stddev_overall = df_long.groupby(["common_cause_id"]).agg(stddev=("W", np.std)).reset_index()
        mean_diff_overall = pd.merge(mean_diff_overall, stddev_overall, on=["common_cause_id"])
        mean_diff_overall["std_mean_diff"] = mean_diff_overall["mean_w"] / mean_diff_overall["stddev"]

        # Third, concatenating them and plotting
        mean_diff_overall = mean_diff_overall[["common_cause_id", "std_mean_diff"]]
        mean_diff_strata["sample"] = "PropensityAdjusted"
        mean_diff_overall["sample"] = "Unadjusted"
        plot_df = pd.concat([mean_diff_overall, mean_diff_strata])

        import matplotlib.pyplot as plt

        try:
            plt.style.use("seaborn-white")  # For older matplotlib
        except OSError:
            plt.style.use("seaborn-v0_8-white")  # For matplotlib 3.6+
        fig, ax = plt.subplots(1, 1)
        for label, subdf in plot_df.groupby("common_cause_id"):
            subdf.plot(kind="line", x="sample", y="std_mean_diff", ax=ax, label=label)
        plt.legend(title="Common causes")
        plt.ylabel("Standardized mean difference between treatment and control")
        plt.xlabel("")
        plt.xticks(rotation=45)
        return plot_df
