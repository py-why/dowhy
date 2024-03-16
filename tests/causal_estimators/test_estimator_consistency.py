import statsmodels as sm
from pytest import mark

from dowhy.causal_estimators.generalized_linear_model_estimator import GeneralizedLinearModelEstimator
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_estimators.propensity_score_matching_estimator import PropensityScoreMatchingEstimator
from dowhy.causal_estimators.propensity_score_stratification_estimator import PropensityScoreStratificationEstimator
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator

from .base import SimpleEstimatorWithModelParams


@mark.usefixtures("fixed_seed")
class TestEstimatorConsistency(object):
    @mark.parametrize(
        [
            "Estimator",
            "method_params",
        ],
        [
            (
                PropensityScoreMatchingEstimator,
                {},
            ),
            (
                PropensityScoreStratificationEstimator,
                {},
            ),
            (
                PropensityScoreWeightingEstimator,
                {},
            ),
            (
                LinearRegressionEstimator,
                {},
            ),
            (
                GeneralizedLinearModelEstimator,
                {"glm_family": sm.api.families.Poisson()},
            ),
        ],
    )
    def test_encoding_consistency(
        self,
        Estimator,
        method_params,
    ):
        estimator_tester = SimpleEstimatorWithModelParams(Estimator, method_params)
        estimator_tester.consistent_estimator_encoding_test()
