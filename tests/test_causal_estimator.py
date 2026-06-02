import unittest

import pytest

from dowhy.causal_estimator import CausalEstimator


class MockEstimator(CausalEstimator):
    pass


def test_causal_estimator_placeholder_methods():
    estimator = MockEstimator(None)
    with pytest.raises(NotImplementedError):
        estimator._do(None)
    with pytest.raises(NotImplementedError):
        estimator.construct_symbolic_estimator(None)


class TestCausalEstimator(unittest.TestCase):
    def setUp(self):
        # self.df = pd.read_csv(os.path.join(DATA_PATH,'dgp_1/acic_1_1_data.csv'))
        # self.ate = np.mean(self.df['y1'] - self.df['y0'])
        # treated = self.df[self.df['z']==1]
        # self.att = np.mean(treated['y1'] - treated['y0'])
        self.df = None
        self.ate = 1

    # def test_average_treatment_effect(self):
    #     est_ate = 1
    #     bias = est_ate - self.ate
    #     print(bias)
    #     self.assertAlmostEqual(self.ate, est_ate)

    # def test_average_treatment_effect_on_treated(self):
    #    est_att = 1
    #    self.att=1
    #    bias = est_att - self.att
    #    print(bias)
    #    self.assertAlmostEqual(self.att, est_att)


if __name__ == "__main__":
    unittest.main()


def test_estimate_effect_raises_valueerror_for_missing_estimand():
    """estimate_effect() must raise ValueError when no valid estimand is identified.

    Previously, it silently returned None, giving users no indication of failure.
    See issue #1551 https://github.com/py-why/dowhy/issues/1551
    """
    import dowhy.datasets
    from dowhy import CausalModel

    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=3,
        num_instruments=0,
        num_samples=500,
        treatment_is_binary=True,
    )
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["dot_graph"],
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    with pytest.raises(ValueError, match=r"No valid identified estimand for 'iv'"):
        model.estimate_effect(
            identified_estimand=estimand,
            method_name="iv.instrumental_variable",
        )
